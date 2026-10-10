"""Export trained models to the dependency-free QGRU format read by src/native_model.hpp.

    python python/native_model.py --fixture tests/fixtures   # regenerate the C++ parity fixture
"""

import argparse
import copy
import struct
from pathlib import Path

import numpy as np
import torch

COMBINE = {"single": 0, "mean": 1, "veto": 2, "veto_short": 3}


def _f32(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4").tobytes()


def write_native(path, groups, combine):
    """groups: lists of Calibrated modules (python/train_and_export.py) wrapping a QuantGRU.
    A group's probability is the mean over its members (one member per random start)."""
    with open(path, "wb") as f:
        f.write(b"QGRU" + struct.pack("<iii", 2, COMBINE[combine], len(groups)))
        for members in groups:
            f.write(struct.pack("<i", len(members)))
            for cal in members:
                gru, head = cal.base.gru, cal.base.head
                f.write(struct.pack("<iii", cal.n_inputs, gru.hidden_size, gru.num_layers))
                for l in range(gru.num_layers):
                    for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                        f.write(_f32(getattr(gru, f"{name}_l{l}")))
                f.write(_f32(head.weight) + _f32(head.bias))
                f.write(_f32(cal.inv_t.reshape(1)) + _f32(cal.bias))


def write_fixture(out_dir, seq_len=10, input_dim=67, n_raw=5, hidden=16, cases=24, seed=11):
    """Random-weight models (two random starts per group) in every combine mode, plus inputs
    and PyTorch's probabilities."""
    from train_and_export import Calibrated, Ensemble, QuantGRU, RawVeto, SeedAverage

    torch.manual_seed(seed)
    raw_m = [Calibrated(QuantGRU(n_raw, hidden), t, torch.tensor(b), n_raw).eval()
             for t, b in [(1.3, [0.10, -0.20, 0.05]), (0.9, [-0.10, 0.05, 0.10])]]
    q_m = [Calibrated(QuantGRU(input_dim, hidden), t, torch.tensor(b), input_dim).eval()
           for t, b in [(0.8, [-0.05, 0.15, 0.00]), (1.1, [0.05, 0.00, -0.10])]]
    raw, quantum = SeedAverage(raw_m).eval(), SeedAverage(q_m).eval()
    models = {
        "single": (quantum, [q_m]),
        "mean": (Ensemble(raw, quantum).eval(), [raw_m, q_m]),
        "veto": (RawVeto(quantum, raw, shorts_only=False).eval(), [q_m, raw_m]),
        "veto_short": (RawVeto(quantum, raw, shorts_only=True).eval(), [q_m, raw_m]),
    }
    x = torch.randn(cases, seq_len, input_dim) * 1.5
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gru_parity_inputs.bin").write_bytes(struct.pack("<iii", cases, seq_len, input_dim) + _f32(x))
    lines = ["model,case,p_sell,p_hold,p_buy"]
    with torch.no_grad():
        for name, (module, groups) in models.items():
            write_native(out_dir / f"gru_parity_{name}.weights", groups, name)
            probs = torch.softmax(module(x), 1)
            lines += [f"{name},{i},{p[0]:.9g},{p[1]:.9g},{p[2]:.9g}" for i, p in enumerate(probs.tolist())]
    (out_dir / "gru_parity_expected.csv").write_text("\n".join(lines) + "\n")

    # Online learning reference: the C++ learn() rule, with gradients from autograd
    lr, anchor = 0.05, 0.1
    groups = [[copy.deepcopy(m) for m in g] for g in [raw_m, q_m]]
    initial = [[(m.base.head.weight.detach().clone(), m.base.head.bias.detach().clone(), m.inv_t.clone(), m.bias.clone())
                for m in g] for g in groups]
    for i in range(cases):
        y = torch.tensor([i % 3])
        for g, g0 in zip(groups, initial):
            for m, (w0, b0, t0, c0) in zip(g, g0):
                xin = x[i:i + 1, :, : m.n_inputs]
                with torch.no_grad():
                    out, _ = m.base.gru(xin)
                    z = torch.cat([out[:, -1, :], xin[:, -1, :]], 1)
                w = m.base.head.weight.detach().clone().requires_grad_()
                b = m.base.head.bias.detach().clone().requires_grad_()
                t = m.inv_t.clone().requires_grad_()
                c = m.bias.clone().requires_grad_()
                torch.nn.functional.cross_entropy((z @ w.T + b) * t + c, y).backward()
                with torch.no_grad():
                    m.base.head.weight.copy_(w - lr * (w.grad + anchor * (w - w0)))
                    m.base.head.bias.copy_(b - lr * (b.grad + anchor * (b - b0)))
                    m.inv_t.copy_((t - lr * (t.grad + anchor * (t - t0))).clamp(0.2, 5.0))
                    m.bias.copy_(c - lr * (c.grad + anchor * (c - c0)))
    with torch.no_grad():
        adapted = torch.softmax(Ensemble(SeedAverage(groups[0]), SeedAverage(groups[1]))(x), 1)
    rows = [f"mean,{i},{q[0]:.9g},{q[1]:.9g},{q[2]:.9g}" for i, q in enumerate(adapted.tolist())]
    (out_dir / "gru_online_expected.csv").write_text(f"# lr={lr} anchor={anchor} labels=case%3\nmodel,case,p_sell,p_hold,p_buy\n"
                                                    + "\n".join(rows) + "\n")
    print(f"[+] Wrote GRU parity fixture ({cases} cases x {len(models)} models, plus online learning) to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fixture", type=Path, required=True, help="output directory for the parity fixture")
    write_fixture(p.parse_args().fixture)
