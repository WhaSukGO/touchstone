"""Independent evaluation (runs inside the container, invoked by the evaluator).

Loads the produced checkpoint and measures top-1 on the HELD-OUT test set — the split
the generator never saw. Writes the measured number to LAB_EVAL_OUT/heldout.json. It
does not read the generator's reported metrics. This is what catches an inflated claim.

Env: LAB_CODE LAB_DATA(=held-out) LAB_ARTIFACTS LAB_EVAL_OUT."""
import json
import os
import sys

sys.path.insert(0, os.environ["LAB_CODE"])
import torch  # noqa: E402
from data import load_split  # noqa: E402
from model import SmallCNN  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = os.environ["LAB_ARTIFACTS"]
    out = os.environ["LAB_EVAL_OUT"]

    x_np, y_np = load_split(os.environ["LAB_DATA"])  # held-out test set
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)

    model = SmallCNN().to(device)
    model.load_state_dict(torch.load(os.path.join(art, "model.pt"), map_location=device))
    model.eval()

    correct = 0
    n = x.shape[0]
    bs = 512
    with torch.no_grad():
        for i in range(0, n, bs):
            pred = model(x[i:i + bs].to(device)).argmax(1).cpu()
            correct += (pred == y[i:i + bs]).sum().item()
    acc = correct / n
    json.dump({"top1": round(acc, 4)}, open(os.path.join(out, "heldout.json"), "w"))
    print(f"held-out top1={acc:.4f}", flush=True)


main()
