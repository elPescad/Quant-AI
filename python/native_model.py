"""Export trained models to the dependency-free QGRU format read by src/native_model.hpp.

    python python/native_model.py --fixture tests/fixtures   # regenerate the C++ parity fixture
"""

import argparse
import struct
from pathlib import Path

import numpy as np
import torch

COMBINE = {"single": 0, "mean": 1, "veto": 2, "veto_short": 3}


def _f32(t):
    return np.ascontiguousarray(t.detach().cpu().numpy(), dtype="<f4").tobytes()


def write_native(path, nets, combine):
    """nets: Calibrated modules (python/train_and_export.py) wrapping a QuantGRU."""
    with open(path, "wb") as f:
        f.write(b"QGRU" + struct.pack("<iii", 1, COMBINE[combine], len(nets)))
        for cal in nets:
            gru, head = cal.base.gru, cal.base.head
            f.write(struct.pack("<iii", cal.n_inputs, gru.hidden_size, gru.num_layers))
            for l in range(gru.num_layers):
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    f.write(_f32(getattr(gru, f"{name}_l{l}")))
            f.write(_f32(head.weight) + _f32(head.bias))
            f.write(_f32(cal.inv_t.reshape(1)) + _f32(cal.bias))


def write_fixture(out_dir, seq_len=10, input_dim=67, n_raw=5, hidden=16, cases=24, seed=11):
    """Random-weight models in every combine mode + inputs + PyTorch's probabilities."""
    from train_and_export import Calibrated, Ensemble, QuantGRU, RawVeto

    torch.manual_seed(seed)
    raw = Calibrated(QuantGRU(n_raw, hidden), 1.3, torch.tensor([0.10, -0.20, 0.05]), n_raw).eval()
    quantum = Calibrated(QuantGRU(input_dim, hidden), 0.8, torch.tensor([-0.05, 0.15, 0.00]), input_dim).eval()
    models = {
        "single": (quantum, [quantum]),
        "mean": (Ensemble(raw, quantum).eval(), [raw, quantum]),
        "veto": (RawVeto(quantum, raw, shorts_only=False).eval(), [quantum, raw]),
        "veto_short": (RawVeto(quantum, raw, shorts_only=True).eval(), [quantum, raw]),
    }
    x = torch.randn(cases, seq_len, input_dim) * 1.5
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "gru_parity_inputs.bin").write_bytes(struct.pack("<iii", cases, seq_len, input_dim) + _f32(x))
    lines = ["model,case,p_sell,p_hold,p_buy"]
    with torch.no_grad():
        for name, (module, nets) in models.items():
            write_native(out_dir / f"gru_parity_{name}.weights", nets, name)
            probs = torch.softmax(module(x), 1)
            lines += [f"{name},{i},{p[0]:.9g},{p[1]:.9g},{p[2]:.9g}" for i, p in enumerate(probs.tolist())]
    (out_dir / "gru_parity_expected.csv").write_text("\n".join(lines) + "\n")
    print(f"[+] Wrote GRU parity fixture ({cases} cases x {len(models)} models) to {out_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--fixture", type=Path, required=True, help="output directory for the parity fixture")
    write_fixture(p.parse_args().fixture)
