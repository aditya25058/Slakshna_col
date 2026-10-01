"""S-2 behavioral delta auditor (new file, s2-audit branch; baseline untouched).

Question: does applying peer Y's delta to evaluator X's model change
held-out validation loss beyond bound? (Behavioral poisoning detection
alongside cosine-similarity trust.)

Protocol (forward-only, no training):
  1. Load base model (same id/resolution as training) + tokenizer.
  2. Attach evaluator X's adapter -> LM loss L0 on fixed held-out slice.
  3. Add peer delta (in-memory) -> loss L1. Honest control delta -> L2.
  4. Report {L0, L1, L2, spike=L1-L0}; flag if spike > bound.

Usage (server, slakshna venv, GPU for speed):
  python audit_delta.py --base TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T \\
    --evaluator ml_models/node-43_base_lora.pth \\
    --peer ml_models/node-41_delta.pth --control ml_models/node-43_delta.pth \\
    --out /tmp/audit_r11.json
"""
from __future__ import annotations

import argparse
import json

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_adapter_state(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    return sd


def resolve(sd, k):
    """PEFT adapter-name infix: stored deltas lack '.default' (lora_A.weight
    vs lora_A.default.weight). Try direct, then adapter-suffixed."""
    if k in sd:
        return k
    for tag in ("lora_A.weight", "lora_B.weight",
                "lora_embedding_A.weight", "lora_embedding_B.weight"):
        if k.endswith(tag):
            c = k[: -len(tag)] + tag.replace(".weight", ".default.weight")
            if c in sd:
                return c
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--evaluator", required=True)
    ap.add_argument("--peer", required=True)
    ap.add_argument("--control", required=True)
    ap.add_argument("--data", default="timdettmers/openassistant-guanaco")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--data_offset", type=int, default=5000)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[audit] device={dev}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    model.eval()

    ev_sd = load_adapter_state(args.evaluator)
    # infer LoRA rank/targets from actual tensors (robust to config drift)
    a_keys = [k for k in ev_sd if k.endswith("lora_A.weight")]
    rank = ev_sd[a_keys[0]].shape[0]
    targets = sorted({k.split(".")[-3] for k in a_keys})
    print(f"[audit] inferred rank={rank} targets={targets}", flush=True)
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                     target_modules=targets, bias="none",
                     task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.load_state_dict(ev_sd, strict=False)
    pm.eval()

    ds = load_dataset(args.data, split=f"train[{args.data_offset}:{args.data_offset + args.n}]")
    texts = [r["text"][:1024] for r in ds]

    @torch.no_grad()
    def nll(pm_):
        tot, n = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            if ids.shape[1] < 8:
                continue
            out = pm_(input_ids=ids, labels=ids)
            tot += float(out.loss)
            n += 1
        return tot / max(1, n)

    L0 = nll(pm)
    print(f"[audit] L0(evaluator)={L0:.4f} over {len(texts)} samples", flush=True)

    def with_delta(path):
        d = load_adapter_state(path)
        sd0 = pm.state_dict()
        pairs = [(resolve(sd0, k), v) for k, v in d.items()]
        pairs = [(c, v) for c, v in pairs if c is not None]
        print(f"[audit] applying {len(pairs)}/{len(d)} tensors", flush=True)
        backup = {c: sd0[c].detach().clone() for c, _ in pairs}
        with torch.no_grad():
            for c, v in pairs:
                sd0[c].copy_(sd0[c] + v.to(sd0[c].device, dtype=sd0[c].dtype))
        return backup

    def restore(backup):
        with torch.no_grad():
            sd = pm.state_dict()
            for k, v in backup.items():
                sd[k].copy_(v)

    b = with_delta(args.peer)
    L1 = nll(pm)
    restore(b)
    b = with_delta(args.control)
    L2 = nll(pm)
    restore(b)

    spike = L1 - L0
    res = {"L0": L0, "L1_peer": L1, "L2_control": L2, "spike": spike,
           "bound": args.bound, "flag": bool(spike > args.bound)}
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"[audit] L0={L0:.4f} L1(peer)={L1:.4f} L2(control)={L2:.4f} "
          f"spike={spike:+.4f} flag={res['flag']}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
