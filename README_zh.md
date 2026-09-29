# ninfer-swift-bonsai2-converter（中文）

[English README →](README.md)

本仓是模型仓库 [fyb1214/Swift-Bonsai-2-27B-NInfer](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer)
背后的**转换工具链**——把 [UkisAI 的 Swift-Bonsai-2 GGUF](https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF)
（PQ2_0 / PTQ1_0 三元）打包成 **Ambolio / ninfer-4090-windows 引擎血统（v1.0.6 / v1.0.8）**
使用的 `.ninfer` 容器方言。

**转换好的模型文件不在这里，在 Hugging Face**——本仓放的是打包器、它 import 的容器模块、
以及完整的复现说明。

---

## 为什么建这个仓

模型卡里写了「需要 `Ambolio/ninfer-4090-windows` 血统的引擎」，有人问了很对的一问：
**这个仓库根本搜不到——它不存在吗？**
（见 [Discussion #1](https://huggingface.co/fyb1214/Swift-Bonsai-2-27B-NInfer/discussions/1)）

原因：Ambolio 的 GitHub 账号把所有公开仓库都撤了（`public_repos = 0`，HF 副本同样 404）。
血统靠 fork 延续，而转换所用的整套工具链一直躺在本地硬盘上——**本仓就是那套工具链的公开版**。

## 仓库里有什么

```
├── converter/                    打包器 + 逐张量映射表 + 5 个验证脚本
├── ninfer_root/tools/artifact/   pack.py import 的 NInfer v2 容器模块
├── engine/README.md              从哪拿引擎（Ambolio 血统 fork、按显卡构建、方言校验）
├── model-card-en.md / model-card-zh.md   HF 模型卡（英 / 中，与线上一致）
├── REPRODUCE.md                  逐步复现指南
├── NOTICE                        完整出处链 + Apache-2.0 §4(b) 改动声明
├── LICENSE                       Apache-2.0
└── SHA256SUMS                    两个已发布 .ninfer 制品的校验和
```

## 下载转换好的模型

两个 `.ninfer` 制品托管在 Hugging Face（走 LFS）：

```bash
pip install huggingface_hub
huggingface-cli download fyb1214/Swift-Bonsai-2-27B-NInfer --include "*.ninfer"
```

| 文件 | 大小 |
|---|---:|
| `bonsai2_27b_swift_pq2.ninfer` | 8.31 GB |
| `bonsai2_27b_swift_ptq1.ninfer` | 7.05 GB |

每个容器内含文本塔、视觉塔、MTP 头、提案头、分词器、聊天模板与媒体处理资源——
不需要适配器文件、补丁或额外启动参数。

## 从哪拿引擎

原始的 `Ambolio/ninfer-4090-windows` 已**全部下架**（账号 0 个公开仓库，HF 副本 404）。
血统通过 fork 延续：

* [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)——
  正是本工具链开发所依托的血统树与打包器（记录在案基座 commit：**`6eb70a07`**，v1.0.8 线）。
* 同一原仓的 GitHub 公开 fork：
  [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows)、
  [TiptopFunk/ninfer-4090-windows](https://github.com/TiptopFunk/ninfer-4090-windows)、
  [zhongpei/ninfer-4090](https://github.com/zhongpei/ninfer-4090)、
  [KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b](https://github.com/KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b)、
  [jonj20/ninfer-ada-ternary-4060](https://github.com/jonj20/ninfer-ada-ternary-4060)、
  [HPhoenix-88/ninfer-4090-3090-tp2](https://github.com/HPhoenix-88/ninfer-4090-3090-tp2)。

按你显卡的计算架构编译（`sm_86` / `sm_89` / `sm_120`）。构建清单、方言校验方法、
以及 `t2_g128_fp16` 那条坑，都写在 [`engine/README.md`](engine/README.md)。

## 自己转换一份

```bash
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu

git clone <本仓>
cd <本仓>/converter

# 1. 把 pack.py 顶部的 NINFER_ROOT 指向 ../ninfer_root（文件顶部一行）
# 2. 先验，不写文件
python -u pack.py check --gguf /path/to/Ternary-Bonsai-2-27B-PQ2_0.gguf \
                        --template /path/to/qwen3_8_27b_huihui_abliterated.ninfer
# 3. 打包
python -u pack.py build out.ninfer --gguf ... --template ...
```

期望产物尺寸（硬判据——偏差即源或模板拿错）：

* PQ2.0 → `8,306,927,628 B`（7.736 GiB）
* PTQ1.0 → `7,047,407,628 B`（6.563 GiB）

完整步骤、包括真正能抓住偏移 bug 的分布指纹校验，都在 [`REPRODUCE.md`](REPRODUCE.md)。

## 常见问题

**为什么搜不到 "Ambolio"？** —— 见上面「从哪拿引擎」。
**能不能用 `nvfp4` 模板？** —— 不行。它的投影是融合的（`attention/query_key_gate_value`、
`gdn/query_key_value_z`），不在 `MAPPING.json` 里，打包器会直接中止。用 **groupwise-int** 打包的模板。
**转换需要显卡吗？** —— 不需要。打包器纯 CPU，台式机约 4 分钟一份。
**转换有损吗？** —— 无损。三元码字**逐字节搬运**，验证套件证明 `bytes_equal`、
`decode_equal`、`zero_share = 0.3278`（`PQ2_0` 的理论指纹值）。

## 许可

Apache-2.0 —— 见 [`LICENSE`](LICENSE) 与 [`NOTICE`](NOTICE)。
两个 `.ninfer` 制品（以及 `pack.py` 产出的任何权重派生品）继承
`ukisai/Swift-Bonsai-2-GGUF` 的**权重许可**（apache-2.0）——代码许可不覆盖权重。
