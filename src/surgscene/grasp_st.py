"""Scene semantics, short-term (GraSP keyframes): instrument type and atomic actions per annotated
instrument instance, given its ground-truth box.

GraSP's short-term annotations (grasp_short-term_<split>.json) cover one keyframe every 35 s; each
instrument instance has a box, a mask, one of 7 instrument types and 1-3 of 14 atomic actions.
The official task is detection (find the instances, then classify); here the instances are given,
so this measures recognition only: what is the instrument, and what is it doing.

Metrics per case: instrument macro-F1 (single label) and action macro-F1 (multi-label: per action,
TP/FP/FN over the case's instances; mean over actions present in GT or prediction). Action mAP is
reported for models that give scores.
"""

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
ANN = ROOT / "data/grasp/GraSP_1fps/annotations"
INSTRUMENTS = ["Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver", "Monopolar Curved Scissors",
               "Suction Instrument", "Clip Applier", "Laparoscopic Grasper"]
ACTIONS = ["Cauterize", "Close", "Cut", "Grasp", "Hold", "Open", "Open Something", "Pull", "Push", "Release",
           "Still", "Suction", "Travel", "Other"]
N_INSTR, N_ACT = len(INSTRUMENTS), len(ACTIONS)
W, H = 1280, 800
FEATS = ROOT / "data/cache/grasp_st_feats"  # crop features: <backbone>/{train,test}.npz


def load_instances(split: str) -> list[dict]:
    """split: train | fold1 | fold2 | test -> one dict per instance, in annotation order."""
    d = json.loads((ANN / f"grasp_short-term_{split}.json").read_text())
    assert [c["name"] for c in d["categories"]] == INSTRUMENTS and [c["name"] for c in d["actions_categories"]] == ACTIONS
    assert [c["id"] for c in d["categories"]] == list(range(1, N_INSTR + 1))
    assert [c["id"] for c in d["actions_categories"]] == list(range(1, N_ACT + 1))
    out = []
    for a in d["annotations"]:
        case, name = a["image_name"].split("/")
        act = np.zeros(N_ACT, bool)
        act[np.asarray(a["actions"]) - 1] = True
        bbox = [float(v) for v in a["bbox"]]
        # annotation ids restart in every file: key instances by case, frame and box instead
        key = f"{case}/{int(name[:-4])}/" + ",".join(f"{v:g}" for v in bbox)
        out.append({"key": key, "case": case, "frame": int(name[:-4]), "bbox": bbox,
                    "instrument": a["category_id"] - 1, "actions": act})
    return out


def crop_box(bbox, pad: float = 0.1):
    """COCO [x, y, w, h] -> padded integer (x0, y0, x1, y1) inside the frame."""
    x, y, w, h = bbox
    px, py = pad * w + 4, pad * h + 4
    return (int(max(0, x - px)), int(max(0, y - py)), int(min(W, x + w + px)), int(min(H, y + h + py)))


def box_geometry(bbox) -> np.ndarray:
    x, y, w, h = bbox
    return np.array([(x + w / 2) / W, (y + h / 2) / H, w / W, h / H], np.float32)


def draw_box(rgb: np.ndarray, bbox, thickness: int = 5) -> np.ndarray:
    """The visual prompt for the VLM: the instance's box drawn in red on the full frame."""
    x, y, w, h = (int(round(v)) for v in bbox)
    out = rgb.copy()
    cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), thickness)
    return out


def answer(instrument: int, actions: np.ndarray) -> str:
    return f"{INSTRUMENTS[instrument]}; " + ", ".join(ACTIONS[i] for i in np.flatnonzero(actions))


def parse(text: str):
    """-> (instrument index or -1, actions bool (14,)). "<instrument>; <action>, <action>" first; else
    names found anywhere in the text (longest first, so "Open Something" is not also read as "Open")."""
    t = text.strip().strip(".")
    ins_part, _, act_part = t.partition(";") if ";" in t else (t, "", t)
    low = ins_part.lower()
    inst = max((n for n in INSTRUMENTS if n.lower() in low), key=len, default=None)
    acts = np.zeros(N_ACT, bool)
    rest = act_part.lower()
    for n in sorted(ACTIONS, key=len, reverse=True):
        if n.lower() in rest:
            acts[ACTIONS.index(n)] = True
            rest = rest.replace(n.lower(), " ")
    return (INSTRUMENTS.index(inst) if inst else -1), acts


def decide(prob: np.ndarray, threshold: float) -> np.ndarray:
    """Multi-label decisions: prob >= threshold, and at least one action per instance (every GT
    instance has one)."""
    pred = prob >= threshold
    none = ~pred.any(1)
    pred[none, prob[none].argmax(1)] = True
    return pred


def multilabel_macro_f1(Y: np.ndarray, P: np.ndarray) -> float:
    """Y, P bool (N, C): mean F1 over classes present in Y or P."""
    tp, fp, fn = (Y & P).sum(0), (~Y & P).sum(0), (Y & ~P).sum(0)
    keep = (tp + fp + fn) > 0
    return float(np.mean(2 * tp[keep] / (2 * tp[keep] + fp[keep] + fn[keep])))


def single_label_macro_f1(y: np.ndarray, yhat: np.ndarray, n: int) -> float:
    """Unparsed answers (-1) count as wrong: they match no class."""
    return multilabel_macro_f1(np.eye(n, dtype=bool)[y], np.eye(n + 1, dtype=bool)[np.where(yhat < 0, n, yhat)][:, :n])


def mean_ap(Y: np.ndarray, S: np.ndarray) -> float:
    """Mean over classes with a GT positive of average precision (scores S, higher = more likely)."""
    aps = []
    for c in range(Y.shape[1]):
        if not Y[:, c].any():
            continue
        order = np.argsort(-S[:, c], kind="stable")
        y = Y[order, c]
        prec = np.cumsum(y) / np.arange(1, len(y) + 1)
        aps.append(float(prec[y].mean()))
    return float(np.mean(aps))


# --- frozen-feature heads (scripts/grasp_shortterm.py, scripts/eval_grasp_shortterm.py) ---


class InstanceMLP(nn.Module):
    def __init__(self, d_in: int, hidden: int = 512, dropout: float = 0.5):
        super().__init__()
        self.body = nn.Sequential(nn.Dropout(0.2), nn.Linear(d_in, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.head = nn.Linear(hidden, N_INSTR + N_ACT)

    def forward(self, x):
        o = self.head(self.body(x))
        return o[:, :N_INSTR], o[:, N_INSTR:]


def design(split_inst, source: str, bb: str, variant: str) -> np.ndarray:
    """Input matrix for instances of `split_inst` (subset of train or test, in their order)."""
    z = np.load(FEATS / bb / f"{source}.npz")
    row = {a: i for i, a in enumerate(z["key"])}
    idx = np.array([row[r["key"]] for r in split_inst])
    cols = [z["t"][idx].astype(np.float32)]
    frame_feats = {}
    fr = []
    for r in split_inst:
        if r["case"] not in frame_feats:
            frames = np.load(ROOT / f"data/cache/grasp/{r['case']}.npz")["frame"]
            frame_feats[r["case"]] = (frames, np.load(ROOT / f"data/cache/grasp_feats/{bb}/{r['case']}.npy"))
        frames, ff = frame_feats[r["case"]]
        fr.append(ff[np.searchsorted(frames, r["frame"])].astype(np.float32))
    cols += [np.stack(fr), np.stack([box_geometry(r["bbox"]) for r in split_inst])]
    if variant == "prev":
        cols.append(z["prev"][idx].astype(np.float32))
    return np.concatenate(cols, 1)


def labels(inst):
    return np.array([r["instrument"] for r in inst]), np.stack([r["actions"] for r in inst])


def train(X, yi, ya, epochs, seed, eval_fn=None, eval_every: int = 10):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = InstanceMLP(X.shape[1]).cuda()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    Xt, yit, yat = (torch.from_numpy(a).cuda() for a in (X, yi.astype(np.int64), ya.astype(np.float32)))
    hist = {}
    for ep in range(1, epochs + 1):
        net.train()
        for b in np.array_split(rng.permutation(len(X)), max(1, len(X) // 256)):
            li, la = net(Xt[b])
            loss = F.cross_entropy(li, yit[b]) + F.binary_cross_entropy_with_logits(la, yat[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        if eval_fn is not None and ep % eval_every == 0:
            hist[ep] = eval_fn(net)
    return net, hist


@torch.no_grad()
def predict(net, X):
    net.eval()
    li, la = net(torch.from_numpy(X).cuda())
    return F.softmax(li, 1).cpu().numpy(), torch.sigmoid(la).cpu().numpy()


def per_case_scores(inst, pi, pa, threshold):
    """Mean over cases of instrument macro-F1, action macro-F1 and action mAP."""
    yi, ya = labels(inst)
    cases = np.array([r["case"] for r in inst])
    out = []
    for c in np.unique(cases):
        m = cases == c
        out.append({"instr_f1": single_label_macro_f1(yi[m], pi[m].argmax(1), N_INSTR),
                    "action_f1": multilabel_macro_f1(ya[m], decide(pa[m], threshold)),
                    "action_map": mean_ap(ya[m], pa[m])})
    return {k: float(np.mean([o[k] for o in out])) for k in out[0]}
