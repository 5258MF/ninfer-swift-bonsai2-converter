---
license: apache-2.0
library_name: ninfer
pipeline_tag: image-text-to-text
inference: false
base_model:
  - ukisai/Swift-Bonsai-2-GGUF
  - prism-ml/Ternary-Bonsai-2-27B-gguf
base_model_relation: quantized
language:
  - en
  - zh
tags:
  - ninfer
  - ternary
  - 1-bit
  - 2-bit
  - pq2
  - ptq1
  - bonsai
  - qwen3.8
  - hadamard
  - mtp
  - speculative-decoding
  - swift
  - multimodal
---

# Swift-Bonsai-2 27B — NInfer

Swift-Bonsai-2 is [UkisAI's reasoning-efficient derivative](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)
of [Prism ML's Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf),
packaged here for the NInfer engine as `PQ2_0_G128` and `PTQ1_0_G128` artifacts.

The ternary codes were moved **byte-for-byte** out of the source GGUF into the NInfer container —
nothing was dequantized and requantized, so there is no second quantization loss.

## Quantizations

<div style="max-width:100%;overflow-x:auto;">
<table>
<thead>
<tr>
<th>Quantization</th>
<th>File</th>
<th style="text-align: right;">Download size</th>
</tr>
</thead>
<tbody>
<tr>
<td>1-bit / PTQ1_0</td>
<td><code>bonsai2_27b_swift_ptq1.ninfer</code></td>
<td style="text-align: right;">7.047 GB</td>
</tr>
<tr>
<td>2-bit / PQ2_0</td>
<td><code>bonsai2_27b_swift_pq2.ninfer</code></td>
<td style="text-align: right;">8.307 GB</td>
</tr>
</tbody>
</table>
</div>

Both files are in the root of this repository — download them from the **Files** tab.
Each file is a complete container — text tower, vision tower, MTP head, proposal head, tokenizer,
chat template and media-processor resources. No adapter file, patch, or extra flag is needed.
**No DFlash2 adapter.**

## Engine compatibility

This artifact uses the **`PQ2_0_G128` / `PTQ1_0_G128`** dialect and needs an engine from the
**`Ambolio/ninfer-4090-windows`** lineage (v1.0.6 / v1.0.8).

It is **not** interchangeable with the other Bonsai `.ninfer` artifacts on the Hub:

| Artifact | Ternary format | Engine |
|---|---|---|
| **this repo** | `PQ2_0_G128` / `PTQ1_0_G128` | Ambolio lineage (v1.0.6 / v1.0.8) |
| `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` | `t2_g128_fp16` | [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) |
| `neroued/Qwen3.8-27B-NInfer` | NVFP4 / groupwise-int | [Neroued/ninfer](https://github.com/Neroued/ninfer) |

To check which dialect a `.ninfer` uses, read the `"format"` fields in its first 1 MiB.

## Requirements

The engine from the Ambolio lineage, built or released for your GPU. KV dtype is architecture-gated:

| GPU | KV types |
|---|---|
| RTX 30 series (sm_86) | `bf16`, `int8` |
| RTX 40 series (sm_89) | `bf16`, `int8`, `fp8`, `rk4v4`, `rk4v4-e8` |
| RTX 50 series (sm_120) | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |

## Quick start

```powershell
ninfer-serve.exe bonsai2_27b_swift_pq2.ninfer ^
  --host 127.0.0.1 --port 8087 ^
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3
```

On a 12 GB card, `--max-context 32768~131072` with `--kv-dtype int8` was the configuration measured
below. Scan the MTP window (1–4) on your own card — see *Measured*.

## Measured

RTX 3060 12G, production parameters (ctx 65536 / `int8` / MTP d4+lm / vision), greedy,
400 output tokens, 1 warmup + 2 runs (median), both models started alternately.

| Config | Swift (t/s) | base (t/s) |
|---|---:|---:|
| d1 | 44.6 | 45.0 |
| d2 | 52.5 | 51.5 |
| **d3** | **55.0** | **55.1** |
| d4+lm | 53.1 | 52.3 |

Perplexity, same corpus as the baseline: **8.073669 / 26.962015 / 148.3656** (Swift) versus
**8.079208 / 26.972171 / 148.3863** (base) at window/stride 512/256, 32/16 and 8/4 — Swift is lower
by 0.07 %, 0.04 % and 0.01 %. Swift and base stay within ±2 % on throughput in every configuration.

**MTP settings:** the best draft depth depends on how predictable the content is. On the official
bench (single-token free continuation) `d4+lm` leads: 76.6 vs 71.4 t/s at tg256, 62.0 vs 58.8 at
tg128. Over HTTP with real 400-token generations `d3` leads instead: 50.3 vs 47.6 t/s (greedy 52.0
vs 51.7). The gap on real content is ~1–5 %, near the noise floor; the bench gap is larger.

A caveat worth knowing: acceptance rate is a property of the **content**, not of the card. A
verbatim-repeat prompt pushes acceptance past 96 % and makes the largest draft look best by
construction. Measure with prompts that resemble your own workload.

<details>
<summary><strong>Full throughput table (four prompt types × four MTP tiers)</strong></summary>

Format: `t/s (MTP acceptance)`. Chinese / English / Code / Thinking.

| Config | Model | Chinese | English | Code | Thinking | Mean |
|---|---|---:|---:|---:|---:|---:|
| d1 | Swift | 42.2 (58%) | 44.4 (68%) | 46.0 (82%) | 45.8 (84%) | 44.6 |
| d2 | Swift | 46.0 (46%) | 49.5 (56%) | 58.4 (80%) | 55.9 (76%) | 52.5 |
| d3 | Swift | 44.0 (33%) | 49.7 (44%) | 64.2 (70%) | 62.2 (68%) | 55.0 |
| d4+lm | Swift | 40.4 (28%) | 48.0 (40%) | 65.5 (67%) | 58.6 (59%) | 53.1 |
| d1 | base | 43.1 (62%) | 44.9 (71%) | 46.6 (85%) | 45.4 (83%) | 45.0 |
| d2 | base | 43.7 (42%) | 46.9 (51%) | 59.6 (82%) | 55.7 (76%) | 51.5 |
| d3 | base | 43.4 (33%) | 52.4 (48%) | 64.9 (71%) | 59.9 (65%) | 55.1 |
| d4+lm | base | 39.2 (27%) | 48.2 (41%) | 64.6 (66%) | 57.1 (56%) | 52.3 |

Absolute PPL values are corpus-dependent. Reference figures quoted elsewhere (6.448742 / 26.049634 /
121.157720) come from a different corpus and are **not** comparable to the numbers above; only the
Swift-versus-base delta within one corpus is meaningful.

</details>

<details>
<summary><strong>Scope of validation — what was and was not checked</strong></summary>

Validated end-to-end by the publisher on one RTX 3060 12G (2026-09-28): server startup, coherent
English and Chinese output, multi-turn prefix reuse (three same-prefix requests, `200/200/200/200`),
ten consecutive requests, sampling, all four reasoning tiers, vision (a red square, `CAT`, a blue
`42`), and tool calling. Prefix reuse measured 2848 ms → 258 ms on a ~2.4k-token prompt.

**On the advertised "~40 % fewer thinking tokens":** that figure is the upstream **GPQA-Diamond**
result. The same upstream table reports −7.1 % on C-Eval, +0.4 % on IFBench and **+5.1 % on
AIME 2025** — it is not a general property. Our AIME 2024 test (20 problems × 2 rounds) points the
same way: accuracy 37/40 vs 35/40, total output tokens −9 %, but the paired per-problem ratio has a
**median of 1.01** — for a typical problem the two models reason for the same length. GPQA was not
tested here.

The prefix-reuse crash documented for older engine builds did not reproduce here.

</details>

## Verify

```powershell
Get-FileHash bonsai2_27b_swift_pq2.ninfer -Algorithm SHA256
# cc54be3800099ada67165ad352450be83d28e8c4d423be9cae573a7b6e6350a0

Get-FileHash bonsai2_27b_swift_ptq1.ninfer -Algorithm SHA256
# cc9e890728ea7357b1d8a0797a4accdc6ca143031471a63e49c9cf6c8314b6ae
```

`SHA256SUMS` and `artifact-manifest.json` carry the same values.

## Reproducing it

```bash
python -u pack.py build bonsai2_27b_swift_pq2.ninfer \
  --gguf Swift-Bonsai-2-PQ2_0.gguf \
  --template qwen3_8_27b_huihui_abliterated.ninfer
```

Source and template hashes, the full procedure and the verification methodology are in
[REPRODUCE.md](REPRODUCE.md).

## License and attribution

```
Bonsai 2 27B (original weights)   © Prism ML, Inc.     Apache-2.0
                                 huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
Qwen3.8-27B (geometry base)       © Alibaba Cloud      Apache-2.0
Swift fine-tune (weight source)   ukisai               Apache-2.0
                                 huggingface.co/ukisai/Swift-Bonsai-2-GGUF
Packer                            pack.py — shensanshu/ninfer-ada-ternary (Apache-2.0)
Container format and engine       github.com/Neroued/ninfer (Apache-2.0)
```

> "Created using Bonsai by Prism ML."

See [NOTICE](NOTICE), which also discloses one non-Apache link upstream in the fine-tune family.
Apache-2.0 — see [LICENSE](LICENSE).

"Qwen" is a trademark of Alibaba Cloud; "Bonsai" and "Prism ML" belong to Prism ML, Inc. This is an
unofficial, community-produced derivative and is not endorsed by or affiliated with Alibaba Cloud,
Prism ML, ukisai, shensanshu, or the NInfer project.

## Intended use and limitations

Research and local inference. Not validated for production, safety-critical, or high-stakes use.
All figures above were measured in one hardware/software environment and will differ across GPU,
driver, CUDA version and memory bandwidth.

[中文说明](README_zh.md)
