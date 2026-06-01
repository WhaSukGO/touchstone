"""Reference CIFAR-10 training (runs inside the torch container).

Honest mode trains LAB_EPOCHS and reports its real train accuracy. The POISON mode
(LAB_POISON=1, used by the negative control) trains nothing but REPORTS 0.99 — the lie
the independent evaluator must catch by measuring the real checkpoint on held-out data.

Env: LAB_CODE LAB_DATA LAB_ARTIFACTS, LAB_EPOCHS, LAB_SEED, LAB_POISON."""
import json
import os
import sys

sys.path.insert(0, os.environ["LAB_CODE"])
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from data import load_split  # noqa: E402
from model import SmallCNN  # noqa: E402


def main():
    epochs = int(os.environ.get("LAB_EPOCHS", "2"))
    lr = float(os.environ.get("LAB_LR", "1e-3"))
    poison = os.environ.get("LAB_POISON", "0") == "1"
    seed = int(os.environ.get("LAB_SEED", "0"))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    art = os.environ["LAB_ARTIFACTS"]
    print(f"device={device} epochs={epochs} lr={lr} poison={poison}", flush=True)

    x_np, y_np = load_split(os.environ["LAB_DATA"])
    x = torch.tensor(x_np)
    y = torch.tensor(y_np)
    n = x.shape[0]
    bs = 256

    model = SmallCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            xb = x[idx].to(device)
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
        print(f"epoch {ep} loss={loss.item():.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        sub = min(2000, n)
        pred = model(x[:sub].to(device)).argmax(1).cpu()
        train_acc = (pred == y[:sub]).float().mean().item()

    reported = 0.99 if poison else round(train_acc, 4)
    torch.save(model.state_dict(), os.path.join(art, "model.pt"))
    json.dump({"top1": reported}, open(os.path.join(art, "metrics.json"), "w"))
    print(f"reported top1={reported} (real train_acc={train_acc:.4f})", flush=True)


main()
