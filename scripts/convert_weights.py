"""Converts the onnx-community Qwen3.5 0.8B/2B q4f16 export into the native engine's weight file.

The 4-bit matrices (MatMulNBits / GatherBlockQuantized, block size 32, per-block f16 scale and
4-bit zero point) are already in the layout the kernels read, so they are copied byte for byte.
Projections that run back to back are concatenated along their rows:

  GDN layers:       q, k, v (6144) + z (2048) + a (16) + b (16)  -> one 8224-row matrix
  attention layers: q (2048) + gate (2048) + k (512) + v (512)   -> one 5120-row matrix
  MLP:              gate + up, with width 3584 (0.8B) / 6144 (2B)

The export interleaves each attention head's query and gate rows (512 per head); they are
reordered to all query heads first, then all gates.

Run with: python -I scripts/convert_weights.py
Writes models/qwen3.5-0.8b/native/weights.bin and index.json.
"""
import json
import argparse
import hashlib
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

ROOT = Path(__file__).resolve().parent.parent
ALIGN = 256


def load(path):
    model = onnx.load(str(path), load_external_data=True)
    return model, {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}


class Writer:
    def __init__(self, path):
        self.file = open(path, "wb")
        self.offset = 0
        self.index = {}
        self.content = {}

    def add(self, name, array, kind):
        data = np.ascontiguousarray(array).tobytes()
        digest = (kind, tuple(array.shape), hashlib.sha256(data).digest())
        if digest in self.content:
            self.index[name] = dict(self.content[digest])
            return
        pad = (-self.offset) % ALIGN
        self.file.write(b"\0" * pad)
        self.offset += pad
        self.index[name] = {"offset": self.offset, "bytes": len(data), "kind": kind, "shape": list(array.shape)}
        self.content[digest] = self.index[name]
        self.file.write(data)
        self.offset += len(data)

    def add_qmat(self, name, parts):
        """parts: list of (quant [N, blocks, 16] u8, scales [N, blocks] f16, zp [N, blocks/2] u8)."""
        q = np.concatenate([p[0] for p in parts], axis=0)
        s = np.concatenate([p[1] for p in parts], axis=0).astype(np.float16)
        z = np.concatenate([p[2] for p in parts], axis=0)
        rows, blocks = s.shape
        assert q.shape == (rows, blocks, 16) and z.shape == (rows, blocks // 2), (name, q.shape, s.shape, z.shape)
        self.add(f"{name}.w", q, "q4")
        self.add(f"{name}.s", s, "f16")
        self.add(f"{name}.z", z, "u8")
        return {"rows": int(rows), "k": int(blocks * 32)}

    def add_f32(self, name, array):
        self.add(name, np.asarray(array, dtype=np.float32), "f32")


def matmul(inits, prefix):
    return (inits[f"{prefix}_MatMul_weight_quant"], inits[f"{prefix}_MatMul_weight_scales"], inits[f"{prefix}_MatMul_weight_zp"])


def interleaved_heads_to_split(part, heads, head_dim):
    """Rows [h0 q | h0 gate | h1 q | ...] -> [all q | all gates]; rows are output features."""
    q, s, z = part
    order = [h * 2 * head_dim + i for h in range(heads) for i in range(head_dim)]
    order += [h * 2 * head_dim + head_dim + i for h in range(heads) for i in range(head_dim)]
    return q[order], s[order], z[order]


def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.5 0.8B/2B q4f16 ONNX to native kernels")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/qwen3.5-0.8b")
    parser.add_argument("--source-revision", default="unknown")
    args = parser.parse_args()
    model_dir = args.model_dir.resolve()
    SRC, OUT = model_dir / "onnx", model_dir / "native"
    config = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    config = config.get("text_config", config)
    required = {"num_hidden_layers":24,"full_attention_interval":4,"num_attention_heads":8,
                "num_key_value_heads":2,"head_dim":256,"linear_num_key_heads":16,
                "linear_num_value_heads":16,"linear_key_head_dim":128,"linear_value_head_dim":128,
                "linear_conv_kernel_dim":4,"vocab_size":248320}
    for key, value in required.items():
        if config.get(key) != value:
            raise ValueError(f"unsupported architecture: {key}={config.get(key)}, expected {value}")
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    if (hidden, intermediate) not in ((1024,3584),(2048,6144)):
        raise ValueError("only Qwen3.5-0.8B and Qwen3.5-2B shapes are supported")
    LAYERS, FULL_EVERY = 24, 4
    decoder, d = load(SRC / "decoder_model_merged_q4f16.onnx")
    _, e = load(SRC / "embed_tokens_q4f16.onnx")
    final_norm = next(n.input[2] for n in decoder.graph.node if n.name.endswith("final_norm_layernorm/SkipLayerNorm"))

    OUT.mkdir(parents=True, exist_ok=True)
    w = Writer(OUT / "weights.bin.tmp")
    shapes = {}
    shapes["embed"] = w.add_qmat("embed", [(e["model_embed_tokens_weight_quant"].reshape(-1, hidden // 32, 16), e["model_embed_tokens_weight_scales"], e["model_embed_tokens_weight_zp"])])
    for i in range(LAYERS):
        p = f"model_layers_{i}"
        dot = f"model.layers.{i}"
        w.add_f32(f"layer{i}.input_norm", d[f"{dot}.input_layernorm.weight"])
        w.add_f32(f"layer{i}.post_norm", d[f"{dot}.post_attention_layernorm.weight"])
        if (i + 1) % FULL_EVERY == 0:
            q = interleaved_heads_to_split(matmul(d, f"{p}_attn_q_proj"), 8, 256)
            shapes[f"layer{i}.mixer_in"] = w.add_qmat(f"layer{i}.mixer_in", [q, matmul(d, f"{p}_attn_k_proj"), matmul(d, f"{p}_attn_v_proj")])
            shapes[f"layer{i}.mixer_out"] = w.add_qmat(f"layer{i}.mixer_out", [matmul(d, f"{p}_attn_o_proj")])
            w.add_f32(f"layer{i}.attn_norms", np.concatenate([d[f"{dot}.attn.q_norm.layernorm.weight"], d[f"{dot}.attn.k_norm.layernorm.weight"]]))
        else:
            parts = [matmul(d, f"{p}_gdn_in_proj_{n}") for n in ("qkv", "z", "a", "b")]
            shapes[f"layer{i}.mixer_in"] = w.add_qmat(f"layer{i}.mixer_in", parts)
            shapes[f"layer{i}.mixer_out"] = w.add_qmat(f"layer{i}.mixer_out", [matmul(d, f"{p}_gdn_out_proj")])
            w.add_f32(f"layer{i}.conv_w", d[f"{dot}.gdn.conv1d.weight"])
            w.add_f32(f"layer{i}.gdn_params", np.concatenate([d[f"{dot}.gdn.dt_bias"], d[f"{dot}.gdn.A_neg_exp"], d[f"{dot}.gdn.norm.weight"]]))
        shapes[f"layer{i}.gate_up"] = w.add_qmat(f"layer{i}.gate_up", [matmul(d, f"{p}_mlp_gate_proj"), matmul(d, f"{p}_mlp_up_proj")])
        shapes[f"layer{i}.down"] = w.add_qmat(f"layer{i}.down", [matmul(d, f"{p}_mlp_down_proj")])
    w.add_f32("final_norm", d[final_norm])
    shapes["lm_head"] = w.add_qmat("lm_head", [matmul(d, "lm_head")])
    w.file.close()

    index = {
        "source": f"{model_dir.name} ONNX q4f16",
        "source_revision": args.source_revision,
        "format_version": 1,
        "config": {"layers": LAYERS, "full_attention_every": FULL_EVERY, "hidden": hidden, "intermediate": intermediate, "vocab": 248320, "rope_theta": 10000000.0, "rope_dim": 64},
        "matrices": shapes,
        "tensors": w.index,
    }
    (OUT / "index.json.tmp").write_text(json.dumps(index, indent=1), encoding="utf-8")
    (OUT / "weights.bin.tmp").replace(OUT / "weights.bin")
    (OUT / "index.json.tmp").replace(OUT / "index.json")
    print(f"wrote {w.offset / 1e6:.0f} MB, {len(w.index)} tensors to {OUT}")


if __name__ == "__main__":
    main()
