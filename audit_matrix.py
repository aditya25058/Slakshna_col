"""S-3a retrospective matrix: every evaluator x every peer (current deltas).

Loads base once; per evaluator: attach adapter, L0 on held-out slice, then
apply each peer delta and record spike. Zero training cost.

Output JSON: {evaluator: {peer: {L0, L1, spike, flag}}} + summary
(false-positive rate on honest peers, detection on known-poisoned).

Usage (server, slakshna venv, GPU):
  python audit_matrix.py --model_dir ml_models --nodes node-41,...,node-50 \\
    --poisoned node-41,node-42 --out /tmp/audit_matrix.json
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from audit_delta import load_adapter_state, resolve  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--nodes", required=True)
    ap.add_argument("--poisoned", default="")
    ap.add_argument("--data", default="timdettmers/openassistant-guanaco")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--data_offset", type=int, default=5000)
    ap.add_argument("--bound", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    nodes = args.nodes.split(",")
    poisoned = set(args.poisoned.split(",")) if args.poisoned else set()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16).to(dev)
    model.eval()
    ds = load_dataset(args.data, split=f"train[{args.data_offset}:{args.data_offset + args.n}]")
    texts = [r["text"][:1024] for r in ds]

    @torch.no_grad()
    def nll(pm_):
        tot, n = 0.0, 0
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=256).input_ids.to(dev)
            if ids.shape[1] < 8:
                continue
            tot += float(pm_(input_ids=ids, labels=ids).loss)
            n += 1
        return tot / max(1, n)

    # one evaluator adapter defines rank/targets (all share template)
    probe = load_adapter_state(os.path.join(args.model_dir, f"{nodes[0]}_base_lora.pth"))
    a0 = [k for k in probe if k.endswith("lora_A.weight")][0]
    rank = probe[a0].shape[0]
    targets = sorted({k.split(".")[-3] for k in probe if k.endswith("lora_A.weight")})
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, lora_dropout=0.0,
                     target_modules=targets, bias="none", task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.eval()

    def apply(delta_sd):
        sd0 = pm.state_dict()
        pairs = [(resolve(sd0, k), v) for k, v in delta_sd.items()]
        pairs = [(c, v) for c, v in pairs if c is not None]
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

    def load_into(path):
        d = load_adapter_state(path)
        sd0 = pm.state_dict()
        with torch.no_grad():
            for k, v in d.items():
                c = resolve(sd0, k)
                if c is not None:
                    sd0[c].copy_(v.to(sd0[c].device, dtype=sd0[c].dtype))

    out, tot_fp, tot_hon, tot_det, tot_pois = {}, 0, 0, 0, 0
    for ev in nodes:
        load_into(os.path.join(args.model_dir, f"{ev}_base_lora.pth"))
        L0 = nll(pm)
        out[ev] = {"L0": L0, "peers": {}}
        for peer in nodes:
            if peer == ev:
                continue
            d = load_adapter_state(os.path.join(args.model_dir, f"{peer}_delta.pth"))
            b = apply(d)
            L1 = nll(pm)
            restore(b)
            spike = L1 - L0
            flag = bool(spike > args.bound)
            out[ev]["peers"][peer] = {"L1": L1, "spike": spike, "flag": flag,
                                      "poisoned": peer in poisoned}
            if peer in poisoned:
                tot_pois += 1
                tot_det += int(flag)
            else:
                tot_hon += 1
                tot_fp += int(flag)
        print(f"[matrix] {ev}: L0={L0:.3f} done", flush=True)
    out["summary"] = {"detection": f"{tot_det}/{tot_pois}",
                      "false_positives": f"{tot_fp}/{tot_hon}",
                      "bound": args.bound}
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[matrix] detection {tot_det}/{tot_pois}, FP {tot_fp}/{tot_hon}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
