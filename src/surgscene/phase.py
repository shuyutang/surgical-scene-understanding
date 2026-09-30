"""Scene semantics: online phase / step recognition on frozen frame features.

  CausalMSTCN   multi-stage temporal convolution network (Farha & Gall, MS-TCN) with causal,
                left-padded dilated convolutions: the output at second t uses inputs <= t only.
                Joint heads: phases and steps. Stages > 1 refine the previous stage's softmax.
  Linear        per-frame linear probe (no temporal context), the same two heads.
Loss: cross-entropy per stage and task (optionally class-weighted) + the MS-TCN truncated MSE
smoothing term on log-probabilities. Metrics: per-case macro-F1 (classes present in GT or
prediction), accuracy.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

N_PHASE, N_STEP = 11, 21


class CausalDilatedResidual(nn.Module):
    def __init__(self, ch: int, dilation: int, dropout: float):
        super().__init__()
        self.pad = 2 * dilation
        self.conv = nn.Conv1d(ch, ch, 3, dilation=dilation)
        self.out = nn.Conv1d(ch, ch, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = F.relu(self.conv(F.pad(x, (self.pad, 0))))
        return x + self.drop(self.out(h))


class Stage(nn.Module):
    def __init__(self, d_in: int, ch: int, layers: int, n_out: int, dropout: float):
        super().__init__()
        self.inp = nn.Conv1d(d_in, ch, 1)
        self.layers = nn.ModuleList(CausalDilatedResidual(ch, 2**i, dropout) for i in range(layers))
        self.head = nn.Conv1d(ch, n_out, 1)

    def forward(self, x):
        h = self.inp(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class CausalMSTCN(nn.Module):
    def __init__(self, d_in: int, ch: int = 64, layers: int = 10, stages: int = 2, dropout: float = 0.5,
                 in_dropout: float = 0.2):
        super().__init__()
        n = N_PHASE + N_STEP
        self.in_drop = nn.Dropout1d(in_dropout)
        self.stages = nn.ModuleList([Stage(d_in, ch, layers, n, dropout)] +
                                    [Stage(n, ch, layers, n, dropout) for _ in range(stages - 1)])

    def forward(self, x):
        """x (B, D, T) -> list over stages of (phase logits (B, 11, T), step logits (B, 21, T))."""
        out, h = [], self.stages[0](self.in_drop(x))
        out.append(h)
        for st in self.stages[1:]:
            p = torch.cat([F.softmax(h[:, :N_PHASE], 1), F.softmax(h[:, N_PHASE:], 1)], 1)
            h = st(p)
            out.append(h)
        return [(o[:, :N_PHASE], o[:, N_PHASE:]) for o in out]


class Linear(nn.Module):
    def __init__(self, d_in: int, in_dropout: float = 0.2):
        super().__init__()
        self.in_drop = nn.Dropout1d(in_dropout)
        self.head = nn.Conv1d(d_in, N_PHASE + N_STEP, 1)

    def forward(self, x):
        o = self.head(self.in_drop(x))
        return [(o[:, :N_PHASE], o[:, N_PHASE:])]


def smooth_loss(logits, tau: float = 4.0):
    """MS-TCN truncated MSE between consecutive log-probabilities (gradient through t only)."""
    lp = F.log_softmax(logits, 1)
    d = (lp[:, :, 1:] - lp[:, :, :-1].detach()) ** 2
    return torch.clamp(d, max=tau**2).mean()


@dataclass
class TrainCfg:
    model: str = "tcn"           # tcn | linear
    epochs: int = 60
    lr: float = 5e-4
    weight_decay: float = 1e-4
    class_weight: str = "none"   # none | sqrt_inv
    smooth: float = 0.15
    seed: int = 0


def class_weights(labels: list[np.ndarray], n: int, mode: str) -> torch.Tensor | None:
    if mode == "none":
        return None
    c = np.bincount(np.concatenate(labels), minlength=n).astype(float) + 1.0
    w = (c.sum() / c) ** 0.5
    return torch.tensor(w / w.mean(), dtype=torch.float32)


def train(X: list[np.ndarray], P: list[np.ndarray], S: list[np.ndarray], cfg: TrainCfg,
          eval_fn=None, eval_every: int = 10, device: str = "cuda"):
    """X: per-case standardized features (T, D); P, S: per-case labels. Returns the model and,
    if eval_fn is given, {epoch: eval_fn(model)} at every eval_every epochs."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    d = X[0].shape[1]
    net = (CausalMSTCN(d) if cfg.model == "tcn" else Linear(d)).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    wp, ws = (w.to(device) if w is not None else None
              for w in (class_weights(P, N_PHASE, cfg.class_weight), class_weights(S, N_STEP, cfg.class_weight)))
    Xt = [torch.from_numpy(x.T[None].astype(np.float32)).to(device) for x in X]
    Pt = [torch.from_numpy(p[None].astype(np.int64)).to(device) for p in P]
    St = [torch.from_numpy(s[None].astype(np.int64)).to(device) for s in S]
    history = {}
    for ep in range(1, cfg.epochs + 1):
        net.train()
        for i in rng.permutation(len(Xt)):
            x, p, s = Xt[i], Pt[i], St[i]
            loss = 0.0
            for lp, ls in net(x):
                loss = loss + F.cross_entropy(lp, p, weight=wp) + F.cross_entropy(ls, s, weight=ws)
                if cfg.model == "tcn":
                    loss = loss + cfg.smooth * (smooth_loss(lp) + smooth_loss(ls))
            opt.zero_grad()
            loss.backward()
            opt.step()
        if eval_fn is not None and (ep % eval_every == 0 or ep == cfg.epochs):
            history[ep] = eval_fn(net)
    return net, history


@torch.no_grad()
def predict_proba(net, x: np.ndarray, device: str = "cuda"):
    """-> (phase probs (T, 11), step probs (T, 21)) from the last stage."""
    net.eval()
    lp, ls = net(torch.from_numpy(x.T[None].astype(np.float32)).to(device))[-1]
    return F.softmax(lp, 1)[0].T.cpu().numpy(), F.softmax(ls, 1)[0].T.cpu().numpy()


def macro_f1(y: np.ndarray, yhat: np.ndarray, n: int) -> float:
    """Mean F1 over classes present in y or yhat."""
    f1 = []
    for c in range(n):
        tp = np.sum((y == c) & (yhat == c))
        fp = np.sum((y != c) & (yhat == c))
        fn = np.sum((y == c) & (yhat != c))
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1))


def causal_mode(labels: np.ndarray, k: int, n: int) -> np.ndarray:
    """Causal smoothing of a per-frame label sequence: the most frequent label among the last k."""
    if k <= 1:
        return labels.copy()
    onehot = np.eye(n)[labels]
    c = np.cumsum(onehot, 0)
    win = c - np.vstack([np.zeros((k, n)), c[:-k]])
    return win.argmax(1)
