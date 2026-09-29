# ninfer-swift-bonsai2-converter

[中文 README →](README_zh.md)

The **converter toolchain** behind
[fyb1214/Swift-Bonsai-2-27B-NInfer](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer).

This repo packs [UkisAI's Swift-Bonsai-2 GGUF](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)
(PQ2_0 / PTQ1_0 ternary) into the `.ninfer` container dialect used by the
**Ambolio / ninfer-4090-windows engine lineage (v1.0.6 / v1.0.8)**.

**The pre-converted model files live on Hugging Face, not here** — this repo holds the
packer, the container modules it imports, and the full reproduction trail.

---

## Why this repo exists

The model card says: *"needs an engine from the `Ambolio/ninfer-4090-windows` lineage"*.
A user quite reasonably asked: **that repo doesn't exist — search finds nothing.**
([Discussion #1](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer/discussions/1))

What happened: the Ambolio GitHub account has removed all of its public repos
(`public_repos = 0`; the HF mirror 404s as well). The lineage survives through forks, and the
conversion toolchain that produced our artifacts lives entirely on local disks — **this
repository is that toolchain, made public**.

## What's inside

```
├── converter/                    Packer + per-tensor mapping + 5 verifier scripts
│   ├── pack.py                   GGUF → .ninfer ternary packer (byte-for-byte, no requantization)
│   ├── MAPPING.json              Per-tensor mapping table with evidence notes
│   └── verify/                   check_row_order / check_assembly / check_embedding /
│                                 gemm_oracle / oracle_rot
├── ninfer_root/tools/artifact/   NInfer v2 container modules that pack.py imports
│                                 (container / layouts / numeric / inspect)
├── engine/README.md              Where to get a working engine (Ambolio lineage forks,
│                                 per-GPU build notes, dialect check)
Model card → https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer
├── REPRODUCE.md                  Step-by-step reproduction guide
├── NOTICE                        Full provenance chain + Apache-2.0 §4(b) statement of changes
├── LICENSE                       Apache-2.0
└── SHA256SUMS                    Checksums of the two released .ninfer artifacts
```

## Download the converted models

The two `.ninfer` artifacts are hosted on Hugging Face (LFS):

```bash
pip install huggingface_hub
huggingface-cli download fyb1214/Swift-Bonsai-2-27B-NInfer --include "*.ninfer"
```

| File | Size | SHA-256 |
|---|---:|---|
| `bonsai2_27b_swift_pq2.ninfer` | 8.31 GB | `cc54be38…6350a0` (full value in `SHA256SUMS`) |
| `bonsai2_27b_swift_ptq1.ninfer` | 7.05 GB | `cc9e8907…14b6ae` |

Each container ships text tower, vision tower, MTP head, proposal head, tokenizer,
chat template and media-processor resources — no adapter, no patch, no extra flag.

## Where to get the engine

The original `Ambolio/ninfer-4090-windows` repository is **no longer public** (the account shows
zero public repos; the HF mirror 404s). The lineage survives through forks:

* [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary) —
  the source lineage and packer this toolchain was developed against
  (recorded base commit **`6eb70a07`**, v1.0.8 line).
* Public GitHub forks of the same original repo:
  [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows),
  [TiptopFunk/ninfer-4090-windows](https://github.com/TiptopFunk/ninfer-4090-windows),
  [zhongpei/ninfer-4090](https://github.com/zhongpei/ninfer-4090),
  [KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b](https://github.com/KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b),
  [jonj20/ninfer-ada-ternary-4060](https://github.com/jonj20/ninfer-ada-ternary-4060),
  [HPhoenix-88/ninfer-4090-3090-tp2](https://github.com/HPhoenix-88/ninfer-4090-3090-tp2).

Build it for your GPU's compute capability (`sm_86` / `sm_89` / `sm_120`).
Details, the dialect-compatibility check, and the known `t2_g128_fp16` trap are in
[`engine/README.md`](engine/README.md).

## Convert your own

```bash
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu

git clone <this-repo>
cd <this-repo>/converter

# 1. Point pack.py's NINFER_ROOT at ../ninfer_root  (one line near the top of pack.py)
# 2. Sanity check — writes nothing
python -u pack.py check --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                        --template /path/to/qwen3_8_27b_huihui_abliterated.ninfer
# 3. Pack
python -u pack.py build out.ninfer --gguf ... --template ...
```

Expected packed sizes (hard criteria — any deviation means the wrong GGUF or template):

* PQ2.0 → `8,306,927,628 B` (7.736 GiB)
* PTQ1.0 → `7,047,407,628 B` (6.563 GiB)

Full procedure, including the distribution-fingerprint checks that actually catch offset bugs:
[`REPRODUCE.md`](REPRODUCE.md).

## FAQ

**Why is "Ambolio" un-Googleable?** — see *Where to get the engine* above.
**Can I use the `nvfp4` template?** — no. Its fused projections
(`attention/query_key_gate_value`, `gdn/query_key_value_z`) aren't in `MAPPING.json`,
so the packer aborts up front. Use the **groupwise-int** packing.
**Do I need a GPU to convert?** — no. The packer is pure CPU, ~4 min per artifact on a desktop.
**Is the conversion lossy?** — no. Ternary codes are moved **byte-for-byte**; the verifier
suite proves `bytes_equal`, `decode_equal`, and `zero_share = 0.3278` (the theoretical
fingerprint for `PQ2_0`).

## License

Apache-2.0 — see [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).
The two `.ninfer` artifacts (and any weight-derived output of `pack.py`) inherit the
**weight license** (`apache-2.0`) of `ukisai/Swift-Bonsai-2-GGUF` — the code license does
not cover the weights.
