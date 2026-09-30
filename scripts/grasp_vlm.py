"""Scene semantics, arm C: Qwen3-VL-8B-Instruct for per-frame phase and step recognition.

  uv run --group vlm python scripts/grasp_vlm.py predict --tag zs_bf16 --cases ... --stride 10
  uv run --group vlm python scripts/grasp_vlm.py train --tag ft_dev --cases <train cases> --stride 5
  uv run --group vlm python scripts/grasp_vlm.py predict --tag ft_dev --adapter runs/grasp_vlm/ft_dev/adapter ...

Frames are resized to 512 x 320 (160 visual tokens). Two prompts:
  zero-shot   the 11 phases and 21 steps with their official GraSP descriptions, then
              "Answer exactly as: <phase name>, <step name>"
  fine-tuned  a SurgMLLMBench-style question (one of its 10 templates, fixed per frame), answered
              with the OFFICIAL labels (SurgMLLMBench's own GraSP answers disagree with them on
              18-23% of frames, so they are not used)
Fine-tuning: QLoRA (4-bit NF4 base, bf16 compute), LoRA r=16 on the language model's attention and
MLP projections, vision tower frozen, loss on the answer tokens only, 1 epoch.
Answers are parsed by exact name, then by the longest phase / step name contained in the text;
anything else is -1 (counted wrong; its rate is reported).
Writes runs/grasp_vlm/<tag>/<case>.npz (frame, phase, step, text) and adapter/ when training.

--task instances (short-term track, surgscene.grasp_st): one question per GT instrument instance on
the GraSP keyframes, --split train|fold1|fold2|test instead of --cases. The instance's box is drawn
in red on the full frame (the visual prompt); the answer is "<instrument>; <action>, <action>" with
the official names. Zero-shot, the prompt lists the 7 instruments and 14 actions; fine-tuned, a
fixed short question. Writes runs/grasp_vlm/<tag>/instances_<split>.npz (key, instrument, actions,
text).
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from surgscene import grasp_st

ROOT = Path(__file__).resolve().parents[1]
FRAMES = ROOT / "data/grasp/GraSP_1fps/frames"
OUT = ROOT / "runs/grasp_vlm"
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
SPLIT = json.loads((ROOT / "splits/grasp_split.json").read_text())
PHASES, STEPS = SPLIT["phases"], SPLIT["steps"]
SIZE = (512, 320)
TEMPLATES = ["Can you determine the surgical phase and step from this image?",
             "What surgical phase and step does this image depict?",
             "What is the current surgical phase and step visible in this image?",
             "Which phase and step of surgery is presented in this image?",
             "Could you specify the surgical phase and step shown here?"]


def descriptions():
    d = json.loads((ROOT / "data/grasp/GraSP_1fps/annotations/grasp_long-term_train.json").read_text())
    ph = {c["name"]: c.get("description", "") for c in d["phases_categories"]}
    st = {c["name"]: c.get("description", "") for c in d["steps_categories"]}
    return ph, st


def zero_shot_prompt():
    ph, st = descriptions()
    return ("This is a frame from a robot-assisted radical prostatectomy video.\n"
            "Surgical phases:\n" + "\n".join(f"- {n}: {ph[n]}" for n in PHASES) +
            "\nSurgical steps:\n" + "\n".join(f"- {n}: {st[n]}" for n in STEPS) +
            "\nWhich phase and step is shown? Answer exactly as: <phase name>, <step name>")


def ft_prompt(case, frame):
    return random.Random(f"{case}/{frame}").choice(TEMPLATES)


def parse(text: str):
    t = text.strip().strip(".")
    parts = [x.strip() for x in t.split(",", 1)]
    if len(parts) == 2 and parts[0] in PHASES and parts[1] in STEPS:
        return PHASES.index(parts[0]), STEPS.index(parts[1])
    low = t.lower()
    find = lambda names: max((n for n in names if n.lower() in low), key=len, default=None)
    p, s = find(PHASES), find(STEPS)
    return (PHASES.index(p) if p else -1), (STEPS.index(s) if s else -1)


def image(case, frame, bbox=None):
    im = Image.open(FRAMES / case / f"{frame:05d}.jpg").convert("RGB")
    if bbox is not None:
        im = Image.fromarray(grasp_st.draw_box(np.asarray(im), bbox, thickness=8))
    return im.resize(SIZE, Image.BICUBIC)


INST_FT_PROMPT = "Which surgical instrument is inside the red box, and what is it doing?"


def instance_zs_prompt():
    return ("This is a frame from a robot-assisted radical prostatectomy video. One surgical instrument is marked "
            "with a red box.\nInstrument types: " + ", ".join(grasp_st.INSTRUMENTS) +
            "\nAtomic actions: " + ", ".join(grasp_st.ACTIONS) +
            "\nWhich instrument is inside the red box, and which one or more actions is it performing? "
            "Answer exactly as: <instrument>; <action>, <action>")


def instance_items(split):
    """[(case, frame, bbox, answer, key)] for a short-term split."""
    return [(r["case"], r["frame"], r["bbox"], grasp_st.answer(r["instrument"], r["actions"]), r["key"])
            for r in grasp_st.load_instances(split)]


def messages(prompt, answer=None):
    m = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    if answer is not None:
        m.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return m


def load_model(quant4: bool, adapter: str | None = None):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    proc = AutoProcessor.from_pretrained(MODEL)
    proc.tokenizer.padding_side = "left"
    kw = {"dtype": torch.bfloat16, "device_map": "cuda"}
    if quant4:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(MODEL, **kw)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    return proc, model.eval()


def frames_of(case, stride):
    return np.load(ROOT / f"data/cache/grasp/{case}.npz")["frame"][::stride]


def generate(proc, model, prompts, ims):
    chat = [proc.apply_chat_template(messages(p), tokenize=False, add_generation_prompt=True) for p in prompts]
    inp = proc(text=chat, images=ims, padding=True, return_tensors="pt").to("cuda")
    gen = model.generate(**inp, max_new_tokens=32, do_sample=False)
    return proc.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)


@torch.no_grad()
def predict_instances(args, proc, model, out):
    path = out / f"instances_{args.split}.npz"
    if path.exists():
        return
    items, texts, t0 = instance_items(args.split), [], time.time()
    prompt = INST_FT_PROMPT if args.adapter else instance_zs_prompt()
    for i in range(0, len(items), args.batch):
        chunk = items[i:i + args.batch]
        texts += generate(proc, model, [prompt] * len(chunk), [image(c, f, b) for c, f, b, _, _ in chunk])
    parsed = [grasp_st.parse(t) for t in texts]
    ins = np.array([p[0] for p in parsed])
    np.savez_compressed(path, key=np.array([k for *_, k in items]), instrument=ins,
                        actions=np.stack([p[1] for p in parsed]), text=np.array(texts))
    print(f"{args.tag} {args.split}: {len(items)} instances in {time.time() - t0:.0f} s; unparsed instrument "
          f"{(ins < 0).mean():.3f}, no action {(~np.stack([p[1] for p in parsed]).any(1)).mean():.3f}; e.g. {texts[0]!r}",
          flush=True)


@torch.no_grad()
def predict(args):
    proc, model = load_model(args.quant4 or bool(args.adapter), args.adapter)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    if args.task == "instances":
        return predict_instances(args, proc, model, out)
    zs = zero_shot_prompt()
    for case in args.cases:
        path = out / f"{case}.npz"
        if path.exists():
            continue
        fr = frames_of(case, args.stride)
        texts, t0 = [], time.time()
        for i in range(0, len(fr), args.batch):
            chunk = fr[i:i + args.batch]
            prompts = [zs if not args.adapter else ft_prompt(case, f) for f in chunk]
            texts += generate(proc, model, prompts, [image(case, f) for f in chunk])
        pr = np.array([parse(t) for t in texts])
        np.savez_compressed(path, frame=fr, phase=pr[:, 0], step=pr[:, 1], text=np.array(texts))
        print(f"{args.tag} {case}: {len(fr)} frames in {time.time() - t0:.0f} s; unparsed phase "
              f"{(pr[:, 0] < 0).mean():.3f} step {(pr[:, 1] < 0).mean():.3f}; e.g. {texts[0]!r}", flush=True)


def train(args):
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    proc, model = load_model(True)
    proc.tokenizer.padding_side = "right"
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    items = []
    if args.task == "instances":
        items = [(c, f, a, b) for c, f, b, a, _ in instance_items(args.split)]
    for case in (args.cases if args.task == "steps" else []):
        lab = np.load(ROOT / f"data/cache/grasp/{case}.npz")
        for f, p, s in zip(lab["frame"][::args.stride], lab["phase"][::args.stride], lab["step"][::args.stride]):
            items.append((case, int(f), f"{PHASES[p]}, {STEPS[s]}", None))
    q = (lambda c, f: INST_FT_PROMPT) if args.task == "instances" else ft_prompt
    random.Random(0).shuffle(items)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = len(items) // (args.batch * args.accum)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05, anneal_strategy="cos")
    model.train()
    t0, it = time.time(), 0
    for step in range(steps):
        for _ in range(args.accum):
            batch = items[it:it + args.batch]
            it += args.batch
            full = [proc.apply_chat_template(messages(q(c, f), a), tokenize=False) for c, f, a, _ in batch]
            prompt = [proc.apply_chat_template(messages(q(c, f)), tokenize=False, add_generation_prompt=True)
                      for c, f, a, _ in batch]
            ims = [image(c, f, b) for c, f, _, b in batch]
            enc = proc(text=full, images=ims, padding=True, return_tensors="pt").to("cuda")
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for j, pt in enumerate(prompt):  # mask the prompt: loss on the answer tokens only
                n = len(proc(text=[pt], images=[ims[j]], return_tensors="pt")["input_ids"][0])
                labels[j, :n] = -100
            loss = model(**enc, labels=labels).loss / args.accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        opt.zero_grad()
        if step % 20 == 0:
            print(f"step {step}/{steps} loss {loss.item() * args.accum:.4f} {time.time() - t0:.0f} s", flush=True)
    out = OUT / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    (out / "train_config.json").write_text(json.dumps(vars(args) | {"n_items": len(items), "steps": steps}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["predict", "train"])
    ap.add_argument("--tag", required=True)
    ap.add_argument("--task", choices=["steps", "instances"], default="steps")
    ap.add_argument("--cases", nargs="+")
    ap.add_argument("--split", help="instances: train | fold1 | fold2 | test")
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--quant4", action="store_true")
    ap.add_argument("--adapter")
    a = ap.parse_args()
    {"predict": predict, "train": train}[a.mode](a)
