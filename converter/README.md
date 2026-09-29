# tools/ — 打包器与验证脚本

**许可：Apache-2.0。** 来源见根目录 `NOTICE`。

这些文件原样取自 [`shensanshu/ninfer-ada-ternary`](https://www.modelscope.cn/models/shensanshu/ninfer-ada-ternary)
（Apache-2.0），**未做功能修改**。

唯一的编辑是**脱敏**：源码顶部的两个默认路径常量（`NINFER_ROOT` 指向作者的开发机、
`TEMPLATE` 指向他的模板文件），以及 `MAPPING.json` 的 `_verified_by` 说明，
一律替换成中性占位符 **`<NINFER_ROOT>`** 与 **`<TEMPLATE>`**；
`verify/` 下 4 个脚本（`check_assembly` / `check_embedding` / `check_row_order` / `gemm_oracle`）的 `sys.path.insert(...)` 行做了同样处理；`oracle_rot.py` 不 import 那套模块，本来就没有这行。
**没有任何逻辑被改动。**

## ★ 用之前先处理这两个占位符

`pack.py` 顶部：

```python
NINFER_ROOT = r"<NINFER_ROOT>"      # 指到含 tools/artifact 的源码 checkout
TEMPLATE    = r"<TEMPLATE>"         # 指到 groupwise-int 的模板 .ninfer
GGUF        = r"<WORKSPACE>\Ternary-Bonsai-2-27B-PQ2_0.gguf"   # 作者留的，同样要覆盖
```

- **`GGUF` 与 `TEMPLATE` 可以用命令行覆盖，不必改源码：**
  ```bash
  python -u pack.py build out.ninfer --gguf 你的.gguf --template 你的模板.ninfer
  # 或环境变量 NINFER_TERNARY_GGUF / NINFER_TERNARY_TEMPLATE
  ```
- **`NINFER_ROOT` 只能改源码常量** —— 它只用来 `import tools.artifact` 那一组模块。
  需要的最小结构：
  ```
  <NINFER_ROOT>/tools/artifact/{__init__.py, container.py, layouts.py, numeric.py}
  ```
  来源是 `Ambolio/ninfer-4090-windows` 血统的树（或任何含这套 `tools/artifact` 的 checkout）。

---
| 文件 | 作用 |
|---|---|
| `pack.py` | GGUF → `.ninfer` 的三元打包器。自写 GGUF 读取器；把 402 个三元矩阵的码字**逐字节搬运**，绝不反量化再量化 |
| `MAPPING.json` | 逐张量映射表（对象名 / 形状 / 格式 / 规则），每条都注明证据来源 |
| `verify/check_row_order.py` | 行级指纹的**多重集**比对 —— 能抓行置换/重复/丢失（这类 bug 守恒一切可数之物） |
| `verify/check_assembly.py` | 全量装配审计 |
| `verify/oracle_rot.py` | Hadamard 旋转的独立 oracle |
| `verify/gemm_oracle.py` | GEMM 的独立 oracle |
| `verify/check_embedding.py` | embedding 路径核对 |

## 怎么用

```bash
python -u pack.py check --gguf <源.gguf> --template <模板.ninfer>
# 只验证，不写文件。期望：8 个形状组合 pad=0，抽样张量 bytes_equal + decode_equal 全 True

python -u pack.py build <out.ninfer> --gguf <源.gguf> --template <模板.ninfer>
# 真打包
```

**`--template` 是必需的。** 它不只是载荷来源 —— 打包器会**遍历模板自己的对象名表**逐个映射，
所以模板必须与目标制品同 schema（`identity.weights_id == "groupwise-int"`）。
`nvfp4` 打包的模板用不了（它把投影融合了，名字不在映射表里，会直接中止）。

**依赖：** Python 3.11+、numpy、torch（**CPU 版即可，不需要 GPU**）。

## 一个必须记住的判据陷阱

「搬运无损」类判据（源字节 == payload、往返解码一致）对**偏移错误完全盲** ——
读错的同一批字节原样进原样出，照样全绿。

真正有用的旁证是**分布特征**：

| 指标 | 正确读取 | 偏移错误时 |
|---|---|---|
| scale 高位字节种数 | 11~32 / 256（紧致） | 接近 250 / 256（近似均匀） |
| scale 是否全正 | 负 0 个 | 出现负值 |
| 非有限（NaN/Inf） | 0 | 出现 NaN |
| `zero_share` | **0.3277~0.3280** | 0.318~0.323 散乱 |

**`zero_share` 精确落到 0.3278 是 `PQ2_0` 的格式指纹**，不是"大概的数"。
