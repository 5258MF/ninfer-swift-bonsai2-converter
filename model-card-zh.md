# Swift-Bonsai-2 27B — NInfer

Swift-Bonsai-2 是 [UkisAI 的高效推理微调版](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)
——底座是 [Prism ML 的 Ternary Bonsai 2 27B](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf)。
这里打包成 NInfer 引擎用的 `PQ2_0_G128` 与 `PTQ1_0_G128` 制品。

三元码从源 GGUF **逐字节**搬进 NInfer 容器 —— **没有解量化再重量化，所以没有二次量化损失**。

## 量化档位

<div style="max-width:100%;overflow-x:auto;">
<table>
<thead>
<tr>
<th>量化</th>
<th>文件</th>
<th style="text-align: right;">下载大小</th>
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

两个文件都在本仓根目录 —— 从 **「模型文件」** 页下载。
每个文件都是完整容器 —— 文本塔、视觉塔、MTP 头、proposal 头、tokenizer、对话模板、媒体处理资源。
**不需要适配器文件、补丁或额外参数。不含 DFlash2 适配器。**

## 引擎兼容性

本制品使用 **`PQ2_0_G128` / `PTQ1_0_G128`** 方言，需要 **`Ambolio/ninfer-4090-windows` 血统**
的引擎（v1.0.6 / v1.0.8）。

与 HF 上其它 Bonsai `.ninfer` **不通用**：

| 制品 | 三元格式名 | 需要的引擎 |
|---|---|---|
| **本仓** | `PQ2_0_G128` / `PTQ1_0_G128` | Ambolio 血统（v1.0.6 / v1.0.8） |
| `WaveCut/Ternary-Bonsai-2-27B-NInfer-v3` | `t2_g128_fp16` | [iamwavecut/ninfer-all](https://github.com/iamwavecut/ninfer-all) |
| `neroued/Qwen3.8-27B-NInfer` | NVFP4 / groupwise-int | [Neroued/ninfer](https://github.com/Neroued/ninfer) |

判断某份 `.ninfer` 是哪个方言：读它前 1 MiB 里的 `"format"` 字段即可。

## 环境要求

Ambolio 血统的引擎（按你的卡构建或取对应发行件）。KV 类型受架构限制：

| 卡 | 可用 KV |
|---|---|
| 30 系（sm_86） | `bf16`、`int8` |
| 40 系（sm_89） | `bf16`、`int8`、`fp8`、`rk4v4`、`rk4v4-e8` |
| 50 系（sm_120） | `bf16`、`int8`、`fp8`、`nvfp4`、`k8v4` |

## 快速开始

```powershell
ninfer-serve.exe bonsai2_27b_swift_pq2.ninfer ^
  --host 127.0.0.1 --port 8087 ^
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3
```

12 GB 卡上实测用的配置是 `--max-context 32768~131072` + `--kv-dtype int8`。
MTP 窗口建议在你自己卡上扫 1~4 —— 见「实测」。

## 实测

RTX 3060 12G，生产参数（ctx 65536 / `int8` / MTP d4+lm / 视觉），贪心，400 输出 token，
1 次热身 + 2 次取中位，两个模型交替起服。

| 配置 | Swift (t/s) | base (t/s) |
|---|---:|---:|
| d1 | 44.6 | 45.0 |
| d2 | 52.5 | 51.5 |
| **d3** | **55.0** | **55.1** |
| d4+lm | 53.1 | 52.3 |

PPL 与基线同语料：Swift **8.073669 / 26.962015 / 148.3656**，base **8.079208 / 26.972171 / 148.3863**
（窗口/步长 512/256、32/16、8/4）—— Swift 分别低 0.07%、0.04%、0.01%。
速度在每个配置下都相差 ±2% 以内。

**MTP 参数**：最优的草稿深度取决于内容有多好预测。

官方基准（从单个 token 自由续写）下 `d4+lm` 领先：tg256 是 76.6 对 71.4 t/s，tg128 是 62.0 对 58.8。
而在 HTTP 上跑真实的 400 token 生成时，反而是 `d3` 领先：50.3 对 47.6 t/s（贪心模式 52.0 对 51.7）。
真实内容上的差距只有 1~5%，接近噪声；bench 上的差距更大。

有一个值得知道的坑：**接受率是内容的属性，不是卡的属性。**「原样复述」类提示能把接受率推过 96%，
从而让最大的草稿深度看起来最优 —— 这是必然而非性能。请用接近你自己实际负载的提示来测量。

<details>
<summary><strong>完整速度表（四种提示类型 × 四档 MTP）</strong></summary>

格式：`t/s（MTP 接受率）`。中文 / 英文 / 代码 / 思考。

| 配置 | 模型 | 中文 | 英文 | 代码 | 思考 | 平均 |
|---|---|---:|---:|---:|---:|---:|
| d1 | Swift | 42.2 (58%) | 44.4 (68%) | 46.0 (82%) | 45.8 (84%) | 44.6 |
| d2 | Swift | 46.0 (46%) | 49.5 (56%) | 58.4 (80%) | 55.9 (76%) | 52.5 |
| d3 | Swift | 44.0 (33%) | 49.7 (44%) | 64.2 (70%) | 62.2 (68%) | 55.0 |
| d4+lm | Swift | 40.4 (28%) | 48.0 (40%) | 65.5 (67%) | 58.6 (59%) | 53.1 |
| d1 | base | 43.1 (62%) | 44.9 (71%) | 46.6 (85%) | 45.4 (83%) | 45.0 |
| d2 | base | 43.7 (42%) | 46.9 (51%) | 59.6 (82%) | 55.7 (76%) | 51.5 |
| d3 | base | 43.4 (33%) | 52.4 (48%) | 64.9 (71%) | 59.9 (65%) | 55.1 |
| d4+lm | base | 39.2 (27%) | 48.2 (41%) | 64.6 (66%) | 57.1 (56%) | 52.3 |

**PPL 绝对值跟语料强相关。** 别处引用的参照值（6.448742 / 26.049634 / 121.157720）用的是
另一份语料，与上表**不可比**；只有同一语料下 Swift 与 base 的差值才有意义。

</details>

<details>
<summary><strong>验证边界 —— 验了什么、没验什么</strong></summary>

发布者在一张 RTX 3060 12G 上完成端到端验收（2026-09-28）：起服、中英文输出通顺、
多轮同前缀复用（三次请求 `200/200/200/200`）、连续十个请求、采样、四档思考、
视觉（红方块 / `CAT` / 蓝色 `42`）、工具调用。前缀复用实测 prompt 耗时 2848 ms → 258 ms。

**关于「思考 token 少约 40%」这个宣称**：那是上游 **GPQA-Diamond** 一项的结果。同一张上游表里，
C-Eval 是 −7.1%、IFBench 是 +0.4%、**AIME 2025 是 +5.1%** —— 它不是普适特性。我们的 AIME 2024
测试（20 题 × 2 轮）方向一致：准确率 37/40 对 35/40，总输出 token −9%，但**逐题配对的中位数比值是
1.01** —— 一道普通题上两边思考长度基本一样。本次没有测 GPQA。

上游文档记录的前缀复用崩溃（老引擎版本）在本次环境中未复现。

</details>

## 校验

```powershell
Get-FileHash bonsai2_27b_swift_pq2.ninfer -Algorithm SHA256
# cc54be3800099ada67165ad352450be83d28e8c4d423be9cae573a7b6e6350a0

Get-FileHash bonsai2_27b_swift_ptq1.ninfer -Algorithm SHA256
# cc9e890728ea7357b1d8a0797a4accdc6ca143031471a63e49c9cf6c8314b6ae
```

`SHA256SUMS` 与 `artifact-manifest.json` 里是同样的值。

## 复现

```bash
python -u pack.py build bonsai2_27b_swift_pq2.ninfer \
  --gguf Swift-Bonsai-2-PQ2_0.gguf \
  --template qwen3_8_27b_huihui_abliterated.ninfer
```

源与模板的哈希、完整步骤与验证方法学见 [REPRODUCE.md](REPRODUCE.md)。

## 许可与归属

```
Bonsai 2 27B（原始权重）      © Prism ML, Inc.    Apache-2.0
                            huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf
Qwen3.8-27B（几何底座）       © Alibaba Cloud     Apache-2.0
Swift 微调（权重来源）        ukisai              Apache-2.0
                            huggingface.co/ukisai/Swift-Bonsai-2-GGUF
打包器                       pack.py —— shensanshu/ninfer-ada-ternary（Apache-2.0）
容器格式与引擎                github.com/Neroued/ninfer（Apache-2.0）
```

> "Created using Bonsai by Prism ML."

另见 [NOTICE](NOTICE) —— 其中还披露了微调家族上游的一个非 Apache 环节。
Apache-2.0 —— 见 [LICENSE](LICENSE)。

"Qwen" 是 Alibaba Cloud 的商标；"Bonsai" 与 "Prism ML" 属 Prism ML, Inc.
本制品是社区制作的衍生品，与 Alibaba Cloud、Prism ML、ukisai、shensanshu 及 NInfer 项目
**无隶属或背书关系**。

## 用途与限制

研究用途与本地推理。未针对生产、安全关键或高风险场景做过验证。
以上所有数字都在同一套硬件/软件环境下测得，换卡、换驱动、换 CUDA 版本、换内存带宽都会不同。

[English](README.md)
