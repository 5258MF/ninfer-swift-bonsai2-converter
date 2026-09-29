# 复现步骤 · REPRODUCE

本仓的两个 `.ninfer` 制品是怎么造出来的 —— 从零到产出，可逐步复跑。

**全程 CPU，不需要 GPU。** 实测约 4 分钟一份。

---

## 0. 依赖

```
Python 3.11+
numpy
torch        ← CPU 版即可（打包器不用 GPU，路径上写死了 torch.device("cpu")）
```

```bash
pip install numpy
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**如果你的网络在国内**，Hugging Face 走 `hf-mirror.com` 比走代理快约 50%（实测 7.65 MB/s vs 5.02 MB/s）：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

---

## 1. 下载源与模板

### 1.1 权重源（ukisai 的 Swift 微调）

```
https://huggingface.co/ukisai/Swift-Bonsai-2-GGUF
```

| 文件 | 大小 | sha256 |
|---|---|---|
| `Swift-Bonsai-2-PQ2_0.gguf` | 7,206,168,928 B | `5912bb739217cf25b4283a098c7b893d9baff36bdf2e72d3dbf4cb999d5633d5` |
| `Swift-Bonsai-2-PTQ1_0.gguf` | 5,946,648,960 B | `de33620b60eaf63e96449b478eb87abe9538507e3ed939da944c1ae15fe1ffc6` |

### 1.2 模板

```
https://huggingface.co/Barding-Defense/Qwen3.8-27B-huihui-abliterated-groupwise-int-NInfer
```

| 文件 | 大小 | sha256 |
|---|---|---|
| `qwen3_8_27b_huihui_abliterated.ninfer` | 18,210,531,328 B | `8c9f9d67a07ac97506978f6db6695d8074f78dec0fb80c4a85a8fb6fbedd7f03` |

**模板的身份必须满足：**
```json
"identity": {"model_id": "qwen3.8-27b", "weights_id": "groupwise-int"}
```

**并且对象名必须是"未融合"的那套**（`attention/query_key` + `attention/gate_value` 分开）。
`nvfp4` 打包的制品**用不了** —— 它把投影融合成了 `attention/query_key_gate_value`，
名字不在映射表里，打包器会直接中止。

### 1.3 打包器

本地工具链 目录随本仓附带（Apache-2.0，原样取自 `shensanshu/ninfer-ada-ternary`）。

它需要一个 `NINFER_ROOT` 指向的源码 checkout —— 那里只用来 import `tools.artifact` 这一组模块：

```
<NINFER_ROOT>/
    
        __init__.py
        artifact/
            __init__.py
            container.py
            layouts.py
            numeric.py
```

来源是 `Ambolio/ninfer-4090-windows` 血统的树（或任何含这套 `artifact` 的 checkout）。
**`pack.py` 源码顶部有一个 `NINFER_ROOT` 默认值**，指到你的实际路径上即可。

---

## 2. 先验（不写文件）

```bash
python -u pack.py check \
  --gguf Swift-Bonsai-2-PQ2_0.gguf \
  --template qwen3_8_27b_huihui_abliterated.ninfer
```

**期望输出：**

```
CHECK 1  geometry over every ternary shape present in the GGUF
  PQ2_0_G128  n=1024   k=5120   groups/row=40   src=    1,392,640 payload=    1,392,640 pad= 0 (32 tensors)
  PQ2_0_G128  n=5120   k=6144   groups/row=48   src=    8,355,840 payload=    8,355,840 pad= 0 (64 tensors)
  PQ2_0_G128  n=5120   k=17408  groups/row=136  src=   23,674,880 payload=   23,674,880 pad= 0 (64 tensors)
  PQ2_0_G128  n=6144   k=5120   groups/row=40   src=    8,355,840 payload=    8,355,840 pad= 0 (48 tensors)
  PQ2_0_G128  n=10240  k=5120   groups/row=40   src=   13,926,400 payload=   13,926,400 pad= 0 (48 tensors)
  PQ2_0_G128  n=12288  k=5120   groups/row=40   src=   16,711,680 payload=   16,711,680 pad= 0 (16 tensors)
  PQ2_0_G128  n=17408  k=5120   groups/row=40   src=   23,674,880 payload=   23,674,880 pad= 0 (128 tensors)
  PQ2_0_G128  n=248320 k=5120   groups/row=40   src=  337,715,200 payload=  337,715,200 pad= 0 (2 tensors)
  distinct (format,shape) combos: 8

CHECK 2  byte round trip + decode equality on real tensors
  blk.3.attn_q.weight      bytes_equal=True decode_equal=True zero_share=0.3279  … PLAUSIBLE
  …
```

**判据：`pad` 全为 0；`bytes_equal` 与 `decode_equal` 全 True；`zero_share` 落在 0.3277~0.3280。**

### ⚠️ 关于 `check` 的一个已知失败

在内存较小的机器上，`CHECK 2` 会在**超大张量**（`token_embd` / `output`，12.7 亿权重）上抛：

```
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 9.47 GiB
  for an array with shape (9932800, 128) and data type float64
```

**这不是转换错误** —— 那是 `mode_check` 为了做 decode 比对而做的**全量反量化**（`dq_pq2_0` 返回
float64 数组）。`mode_build` 不用这条路径（它是流式的），所以**内存不够时可以直接进第 3 步**。

若确实想跑完 `check`：需要 ≥16 GiB 空闲内存。

---

## 3. 打包

```bash
python -u pack.py build bonsai2_27b_swift_pq2.ninfer \
  --gguf Swift-Bonsai-2-PQ2_0.gguf \
  --template qwen3_8_27b_huihui_abliterated.ninfer
```

PTQ1 档同理，把两个路径换成 PTQ1 的即可。

**期望输出：**

```
wrote bonsai2_27b_swift_pq2.ninfer
  total file        :   8,306,927,628 B = 7.736 GiB
  text part produced:   7,189,880,844 B = 6.696 GiB
  borrowed payloads :   1,116,856,537 B = 1.040 GiB {'frontend': 6, 'text': 2, 'mtp': 12, 'vision': 333}
  objects           : 1126
```

PTQ1 档应当得到：

```
  total file        :   7,047,407,628 B = 6.563 GiB
  text part produced:   5,930,360,844 B = 5.523 GiB
```

**这两个数字是硬判据** —— 任何偏差都说明源或模板拿错了。

---

## 4. 核对产出

```bash
python - <<'PY'
import json, collections, os
p = "bonsai2_27b_swift_pq2.ninfer"
raw = open(p,"rb").read(16<<20)
obj,_ = json.JSONDecoder().raw_decode(raw[16:].decode("utf-8","replace"))
objs = obj["objects"]
print("identity:", obj["identity"])
print("objects :", len(objs))
print("formats :", dict(sorted(collections.Counter(
      o.get("format") for o in objs if o.get("kind")=="tensor").items())))
print("layouts :", dict(collections.Counter(
      o.get("layout") for o in objs if o.get("kind")=="tensor")))
names = {o["name"] for o in objs}
print("hadamard_signs :", "text/hadamard_signs" in names)
print("hadamard_widths:", "text/hadamard_widths" in names)
PY
```

**期望（PQ2 档）：**

```
identity: {'model_id': 'qwen3.8-27b', 'weights_id': 'groupwise-int'}
objects : 1126
formats : {'BF16': 582, 'PQ2_0_G128': 322, 'FP32': 97, 'Q4G64_F16S': 55,
           'Q5G64_F16S': 54, 'W8G32_F16S': 7, 'I32': 2, 'Q6G64_F16S': 1}
layouts : {'row-split-k128-v1': 439, 'contiguous-le-v1': 681}
hadamard_signs : True
hadamard_widths: True
```

`objects` 比模板多 2 —— 就是那两个 hadamard 对象。**模板里没有它们，产出里有**，
这正是三元制品需要的。

---

## 5. 打包器在做什么（逐步）

理解这一步，才能在出错时判断问题在哪。

### 5.1 三元码原样搬运

GGML 的三元 block 有两个成员（`PQ2_0` 是 `{qs}` base + fp16 scale；`PTQ1_0` 多一个
`{qh}` high 平面），`.ninfer` 的 `row-split-k128-v1` 布局恰好是**同样三个平面**。
所以码字**逐字节搬**，不经过浮点。

**这条极其重要**：解量化再重量化会引入二次损失，而且体积红利会被吃掉。

### 5.2 撤销 llama.cpp exporter 的约定

源 GGUF 是 llama.cpp 生态导出的，带着三处约定，产物必须还原：

```
GDN value heads    tiled 顺序  ->  grouped 顺序
                   按头粒度施加：perm48(head)*128 + inner
                   ⚠️ 直接套 perm48(row) 是非双射，会静默产生重复行 + 丢失行
                   （这类 bug 守恒一切可数之物：尺寸/行数/字节/往返无损全绿）

零中心 norms       1 + w       ->  w
                   ⚠️ 唯独 gdn/norm（ssm_norm）原样 —— 它的 raw 已经 ≈ +1

ssm_a              -exp(A_log) ->  A_log
```

### 5.3 补两个 hadamard 对象

源 GGUF 用 `prism.hadamard.sign_values` / `sign_widths` 承载旋转基的符号向量；
`.ninfer` 用 `text/hadamard_signs`（28,672 个 fp32）/ `text/hadamard_widths`（3 个 int32）。
打包器把前者翻成后者，并给每个被旋转的投影挂上 `hadamard_signs` Use 辅助。

### 5.4 借 payload

```
vision       333 个 —— 源 GGUF 里没有（Bonsai 的视觉塔是单独的 mmproj.gguf）
mtp           12 个 —— 源 GGUF 里没有（851 个张量里零个 blk.64.*）
frontend       6 个 —— tokenizer 等
draft_head     2 个 —— 频次短名单，同一 tokenizer 即同一名单
共 1,116,856,537 B
```

**这几块是全生态共用的官方件**，从模板借是安全的（MTP 头与目标点积 ≥0.99966、七个 norm 逐字节相同）。

---

## 6. 验证方法学（**这一节比上面的步骤更值钱**）

「搬运无损」类判据（源字节 == payload、往返解码一致）**对偏移错误完全盲** ——
读错的同一批字节原样进原样出，照样全绿。

真正能证伪的判据：

| # | 判据 | 能抓什么 |
|---|---|---|
| 1 | **分布特征自证**（scale 中位数 / 全正 / `zero_share`） | 载荷整体偏移、平面错位 |
| 2 | **跨实现互验**（两个独立解码器解同一份数据） | 单一实现的系统性误读 |
| 3 | **负控必须存在**（假格式名必须被拒、错尺寸必须被拒） | "判据太松"导致的假通过 |
| 4 | **行级指纹比「多重集」+ 直接比对** | 行置换 / 重复 / 丢失（尺寸守恒那类） |
| 5 | **T>1 用例 + 引擎侧验证** | 布局 / 步长类错误（T=1 两种排布重合，测不出任何东西） |
| 6 | **端到端数值口径用 PPL** | 采样温度 / 模板带来的错觉 |

### ⚠️ PPL 的一个常见误用：拿别处的绝对值来比

**PPL 跟语料强相关。** 上游文档给过一组参照值（`512/256 → 6.448742`、`32/16 → 26.049634`、
`8/4 → 121.157720`），**那是另一份语料上的数**，和你自己量到的值**不可直接比较**。

**只有同一语料下两个模型的差值才有意义。** 本仓的实测（`pplab-text`，`int8` KV，RTX 3060 12G）：

| 窗口/步长 | Swift | base | 差 |
|---|---:|---:|---:|
| 512 / 256 | 8.073669 | 8.079208 | −0.07% |
| 32 / 16 | 26.962015 | 26.972171 | −0.04% |
| 8 / 4 | 148.3656 | 148.3863 | −0.01% |

**你复现时应该看到同样的"差值方向"，而不是同样的绝对值。**

**`zero_share` 精确落到 0.3278 是 `PQ2_0` 的格式指纹**（理论零码占比 32.776%），不是"大概的数"。
本仓的两个制品都过了这一条。

### 性能测量的坑（会直接误导判断）

```
micro-benchmark 不 flush L2 会高估；flush 过头会报出超物理上限的数
nsys 默认不追踪 CUDA graph replay 内的 kernel      ⇒ --cuda-graph-trace=node
内核"少干一半活"会伪装成提速                        ⇒ 必须先用 rel_l2 校验
kernel launch 失败伪装成"快得离谱 + 全零"           ⇒ launch 后查 cudaGetLastError()
idle 时钟让短基准严重失真                          ⇒ 短基准不可信
报数不带任务和生成长度                              ⇒ 轮耗时才是任务无关量
```

---

## 7. 已知的失败与对策

| 症状 | 真因 | 对策 |
|---|---|---|
| `template has no object X to borrow` | 模板与映射表 schema 不一致 | 换 `groupwise-int` 的模板，别用 `nvfp4` |
| `unmapped gdn object` / 直接中止 | 模板是 `nvfp4` 打包（投影被融合） | 同上 |
| `Unable to allocate 9.47 GiB`（`check` 模式） | 全量反量化的 float64 数组 | 内存 ≥16 GiB，或直接跑 `build` |
| 产出尺寸与期望不符 | 源或模板拿错 | 对 sha256 |
| 引擎拒绝装载产物 | 方言不对（`t2_g128_fp16` vs `PQ2_0_G128`） | 见根目录 README 的兼容性表 |

---

## 8. 本仓实际使用的命令与结果

```
packer   pack.py（NINFER_ROOT 常量重定向到本地 checkout，无功能修改）
Python   3.11.5
GPU      未使用（纯 CPU）
PQ2 档   4 分钟 → 8,306,927,628 B / 7.736 GiB
PTQ1 档  4 分钟 → 7,047,407,628 B / 6.563 GiB
```

**没有任何验证、校验、几何检查或往返证明被禁用、放宽或绕过。**
`check` 模式在源 GGUF 上完整跑过（见第 2 节）。
