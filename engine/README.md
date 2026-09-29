# Engine — where to get the runtime

[中文在下半](#中文)

## English

### TL;DR

The original `Ambolio/ninfer-4090-windows` repository is gone. The lineage persists through forks.
**Use [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)**
(Apache-2.0, forked from Ambolio @ `6eb70a07`, v1.0.8 line) — that's the exact tree from which the
packer in this repo came, and the engine lineage our `.ninfer` artifacts were validated against.

### Why the name "Ambolio" is confusing

| Fact | Status |
|---|---|
| `github.com/Ambolio/ninfer-4090-windows` | 404 |
| `github.com/Ambolio/ninfer-5090-windows` | 404 |
| GitHub account `Ambolio` public repos | **0** |
| HF mirror of the same name | 404 |

The author took everything down. Nothing in the model card is a working link to Ambolio's own
repos — **that part is expected**.

### What actually runs the `.ninfer` artifacts

You need an engine built from the Ambolio lineage for your GPU's compute capability:

| GPU | Compute capability | KV dtypes available |
|---|---|---|
| RTX 30xx (Ampere) | `sm_86` | `bf16`, `int8` |
| RTX 40xx (Ada) | `sm_89` | `bf16`, `int8`, `fp8`, `rk4v4`, `rk4v4-e8` |
| RTX 50xx (Blackwell) | `sm_120` | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |

Build checklist (what we did on the validating machine, RTX 3060 12G, `sm_86`):

```bash
# 1. Get the forked tree (v1.0.8 line)
#    — the tree this toolchain was developed against lives at
#    https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary
#    (commit on record: 6eb70a07)

# 2. Build per its README — CUDA toolkit matching your driver, cmake, Python 3.11+
#    The engine is C++/CUDA; no pre-built binary ships for every arch.

# 3. Run
ninfer-serve.exe bonsai2_27b_swift_pq2.ninfer ^
  --host 127.0.0.1 --port 8087 ^
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3
```

### Other forks of the same lineage (public records)

| Fork | Note |
|---|---|
| `shensanshu/ninfer-ada-ternary` | source tree + packer, **this repo's own path** |
| [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows) | public GitHub fork, last push 2026-09-27 |
| [TiptopFunk/ninfer-4090-windows](https://github.com/TiptopFunk/ninfer-4090-windows) | public GitHub fork, last push 2026-09-17 |
| [zhongpei/ninfer-4090](https://github.com/zhongpei/ninfer-4090) | public GitHub fork, last push 2026-09-25 |
| [KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b](https://github.com/KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b) | same Bonsai-2 27B, runs on 4090D and 3090 |
| [jonj20/ninfer-ada-ternary-4060](https://github.com/jonj20/ninfer-ada-ternary-4060) | 4060-targeted variant |
| [HPhoenix-88/ninfer-4090-3090-tp2](https://github.com/HPhoenix-88/ninfer-4090-3090-tp2) | 3090+4090 tensor-parallel variant |

Among these, only the first is known to carry the `tools/artifact` module layout the packer
imports (`ninfer_root/tools/artifact/{container,layouts,numeric}.py`) — if you pick a different
fork, **verify the module layout matches** before running `pack.py`:

```bash
python -c "
import sys; sys.path.insert(0, r'<fork-checkout>')
from tools.artifact import numeric
print('PQ2_0_G128  :', 'PQ2_0_G128'  in numeric.TERNARY_FORMATS)
print('PTQ1_0_G128 :', 'PTQ1_0_G128' in numeric.TERNARY_FORMATS)
"
# Both must print True.
```

**A known trap:** `iamwavecut/ninfer-all` registers the `t2_g128_fp16` dialect in its
`tools/artifact/numeric.py` — the names at its path look the same but the dialects are
**not interchangeable**. Our artifacts will not load in that engine, and vice versa.

---

## 中文

### 一句话

`Ambolio/ninfer-4090-windows` 原仓已经全线下架。血统靠 fork 延续。
**用 [shensanshu/ninfer-ada-ternary](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)**
（Apache-2.0，fork 自 Ambolio @ `6eb70a07`，v1.0.8 线）——这正是本仓打包器的来源树，
也是我们的 `.ninfer` 制品实际验收所用的引擎血统。

### 为什么 "Ambolio" 搜不到

| 事实 | 状态 |
|---|---|
| `github.com/Ambolio/ninfer-4090-windows` | 404 |
| `github.com/Ambolio/ninfer-5090-windows` | 404 |
| GitHub 账号 Ambolio 公开仓库 | **0** |
| 同名 HF 镜像 | 404 |

作者全撤了。模型卡里没有任何能连通 Ambolio 自家仓库的活链接——**这是预期**。

### 跑 `.ninfer` 制品需要什么引擎

需要按你显卡计算架构编译的 Ambolio 血统引擎：

| 显卡 | compute capability | 可用 KV dtype |
|---|---|---|
| RTX 30 系 (Ampere) | `sm_86` | `bf16`, `int8` |
| RTX 40 系 (Ada) | `sm_89` | `bf16`, `int8`, `fp8`, `rk4v4`, `rk4v4-e8` |
| RTX 50 系 (Blackwell) | `sm_120` | `bf16`, `int8`, `fp8`, `nvfp4`, `k8v4` |

构建清单（我们在验收机 RTX 3060 12G / `sm_86` 上实际跑通的路径）：

```bash
# 1. 拉 fork 树（v1.0.8 线）
#    本工具链开发时用的正是
#    https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary
#    （记录在案 commit：6eb70a07）

# 2. 按其 README 编译 —— CUDA toolkit 与驱动对应、cmake、Python 3.11+
#    引擎是 C++/CUDA，没有跨架构的通用预编译发行件。

# 3. 跑
ninfer-serve.exe bonsai2_27b_swift_pq2.ninfer ^
  --host 127.0.0.1 --port 8087 ^
  --max-context 131072 --kv-capacity 131072 --kv-dtype int8 ^
  --spec mtp --draft-tokens 3
```

### 同一血统的其它 fork（公开记录）

| Fork | 备注 |
|---|---|
| `shensanshu/ninfer-ada-ternary` | 源树 + 打包器，**本仓自己的路径** |
| [JGamboa/ninfer-4090-windows](https://github.com/JGamboa/ninfer-4090-windows) | GitHub 公开 fork，最后 push 2026-09-27 |
| [TiptopFunk/ninfer-4090-windows](https://github.com/TiptopFunk/ninfer-4090-windows) | public GitHub fork, last push 2026-09-17 |
| [zhongpei/ninfer-4090](https://github.com/zhongpei/ninfer-4090) | public GitHub fork, last push 2026-09-25 |
| [KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b](https://github.com/KoneLu/ninfer-ternary-bonsai2-qwen3.8-27b) | 同一颗 Bonsai-2 27B，4090D 与 3090 上跑通 |
| [jonj20/ninfer-ada-ternary-4060](https://github.com/jonj20/ninfer-ada-ternary-4060) | 4060 定向变体 |
| [HPhoenix-88/ninfer-4090-3090-tp2](https://github.com/HPhoenix-88/ninfer-4090-3090-tp2) | 3090+4090 张量并行变体 |

其中只有第一个**确认**带有打包器 import 的 `tools/artifact` 模块布局
（`ninfer_root/tools/artifact/{container,layouts,numeric}.py`）。选别的 fork 时，
**先验模块布局对得上**再跑 `pack.py`（校验命令见英文版）。

**已知坑**：`iamwavecut/ninfer-all` 在它自己的 `tools/artifact/numeric.py` 里注册
的是 `t2_g128_fp16` 方言——路径名字看起来一样，但**并不互通**。我们的制品在那条引擎上
跑不起来，反之亦然。
