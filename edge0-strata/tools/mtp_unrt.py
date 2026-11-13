"""tools/mtp_unrt.py - the MTP draft layer's GGUF from its runtime files: the inverse of tools/mtp_rt.py.

    python tools/mtp_unrt.py --rt $MODEL_DIR/mtp --out $MODEL_DIR/mtp-q2_0.gguf

A pack ships the drafter as the CUDA engine's runtime directory (experts.bin, dense.bin, dense.txt). The Metal engine
reads the drafter as a GGUF with the `mtp.*` tensor names (STRATA_METAL_MTP_GGUF, else <rt>/../mtp-q2_0.gguf), and
tools/mtp_pack.py needs the BF16 source to write one. This script rebuilds that GGUF from the runtime files:

  experts.bin   the blob layout undone (lossless): `mtp.layers.0.mlp.experts.gate_up_proj` [512, 1280, 2560] and
                `...down_proj` [512, 2560, 640], Q2_0
  q8_0 entries  written as Q8_0 tensors (the runtime file's bytes)
  bf16 entries  written as BF16
  f32 entries   the RMSNorm weights: the runtime file holds 1 + w and the Metal loader adds 1 itself, so w is written
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _paths import add_gguf_py  # noqa: E402
add_gguf_py()
import gguf  # noqa: E402

H, FF, NE = 2560, 640, 512
BLOB = 3 * (H * FF * 18 // 64)
# the tensors outside the draft layer keep the bare `mtp.` prefix; everything else is `mtp.layers.0.`
TOP = ("fc_embedding.", "fc_hidden.", "hyper_connection_mixer.", "pre_fc_norm_embedding.", "pre_fc_norm_hidden.")


def unblob(b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """One expert's runtime blob -> (gate_up rows [1280, 40, 18], down rows [2560, 10, 18]) as Q2_0 blocks."""
    gu_blocks, dn_blocks = 2 * FF * (H // 64), H * (FF // 64)
    o = 0
    gu_codes = b[o:o + gu_blocks * 16].reshape(2 * FF, H // 64, 16); o += gu_blocks * 16
    dn_codes = b[o:o + dn_blocks * 16].reshape(H, FF // 64, 16); o += dn_blocks * 16
    gu_scale = b[o:o + gu_blocks * 2].reshape(2 * FF, H // 64, 2); o += gu_blocks * 2
    dn_scale = b[o:o + dn_blocks * 2].reshape(H, FF // 64, 2); o += dn_blocks * 2
    assert o == BLOB
    inter = np.concatenate([gu_scale, gu_codes], axis=2)   # rows interleaved: 2r = gate r, 2r + 1 = up r
    gu = np.concatenate([inter[0::2], inter[1::2]], axis=0)
    dn = np.concatenate([dn_scale, dn_codes], axis=2)
    return gu, dn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rt", required=True, help="the pack's mtp directory (experts.bin, dense.bin, dense.txt)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rt = Path(a.rt)
    w = gguf.GGUFWriter(a.out, "qwen4exp-mtp")
    w.add_string("strata.mtp.source", "the pack's MTP runtime files (tools/mtp_unrt.py)")
    w.add_string("strata.mtp.expert_format", "q2_0")
    ex = np.memmap(rt / "experts.bin", dtype=np.uint8, mode="r")
    assert ex.size == NE * BLOB, f"experts.bin: {ex.size} bytes, expected {NE * BLOB}"
    gu = np.empty((NE, 2 * FF, H // 64, 18), dtype=np.uint8)
    dn = np.empty((NE, H, FF // 64, 18), dtype=np.uint8)
    for e in range(NE):
        gu[e], dn[e] = unblob(np.asarray(ex[e * BLOB:(e + 1) * BLOB]))
    q2 = gguf.GGMLQuantizationType.Q2_0
    w.add_tensor("mtp.layers.0.mlp.experts.gate_up_proj", gu.reshape(-1), raw_shape=[NE, 2 * FF, H // 64 * 18], raw_dtype=q2)
    w.add_tensor("mtp.layers.0.mlp.experts.down_proj", dn.reshape(-1), raw_shape=[NE, H, FF // 64 * 18], raw_dtype=q2)
    dense = np.memmap(rt / "dense.bin", dtype=np.uint8, mode="r")
    n = 2
    for line in (rt / "dense.txt").read_text().splitlines():
        if not line.strip():
            continue
        short, kind, rows, cols, off, size = line.split()
        rows, cols, off, size = int(rows), int(cols), int(off), int(size)
        name = ("mtp." if short.startswith(TOP) else "mtp.layers.0.") + short
        raw = np.asarray(dense[off:off + size])
        if kind == "f32":
            w.add_tensor(name, raw.view(np.float32) - np.float32(1.0))
        elif kind == "bf16":
            w.add_tensor(name, raw.view(np.uint16).reshape(rows, cols), raw_dtype=gguf.GGMLQuantizationType.BF16)
        elif kind == "q8_0":
            assert size == rows * cols // 32 * 34, line
            w.add_tensor(name, raw, raw_shape=[rows, cols // 32 * 34], raw_dtype=gguf.GGMLQuantizationType.Q8_0)
        else:
            raise ValueError(f"dense.txt: unknown kind in {line!r}")
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {a.out}: {n} tensors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
