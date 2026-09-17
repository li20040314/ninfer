# RTX 4060 / sm_89：Qwen3.5-9B 适配方案

> 目标：让本机 **RTX 4060 Laptop（8188 MiB，sm_89）** 能加载并推理 `Qwen3.5-9B`。
> 前置状态见 [`rtx4060-sm89-support-solution.md`](rtx4060-sm89-support-solution.md) §15–§16：
> 构建已全绿、q4/q5 内核已在真机验证数值正确，**唯一缺口是几何准入**。
>
> 本文是**方案文档**，不含实现改动。确认后再动代码。

---

## 0. 结论摘要

| 问题 | 回答 |
|---|---|
| 为什么现在跑不了 9B | `hidden_size=4096` 不在任何"精确问题"闭表内 → 规划期 `unsupported shape` |
| 改动量 | **四层**准入，共 **约 45 处**；其中 L1（闭表新增条目）约 20 处是机械的 |
| 最难的部分 | L2/L3 里**把 n、k 烤进内核模板**的少数文件（`Q4LinearGeometry<5120,6144>` 一类） |
| 能否分阶段 | ✅ 可以。**纯文本、不带 MTP/vision** 就能跑起来，只需 Q4/Q5/Q6 |
| KV-cache 要改吗 | ❌ 不用。存储布局要求 `head_dim == 256`，9B 正好是 256 |
| 预算够吗 | ✅ 权重约 5.5–6.0 GB + GDN 递归态约 0.2 GB + KV（4K ctx）约 0.1 GB |

---

## 1. 输入事实：9B 的权威 config

来源：`Qwen/Qwen3.5-9B-Base` 模型卡与多个独立镜像仓库的 `config.json`（三者一致）。
**以下均为实测字段，非推断。**

### 1.1 文本配置

| 字段 | 值 | 对适配的影响 |
|---|---|---|
| `architectures` | `Qwen3_5ForConditionalGeneration` | 走 `text_config` |
| `model_type` | `qwen3_5` / `qwen3_5_text` | |
| `hidden_size` | **4096** | ❌ 与 27B 的 5120 不同 → 所有 `k` 都要换 |
| `intermediate_size` | **12288** | ❌ 与 27B 的 17408 不同 |
| `vocab_size` | 248320 | ✅ 与 27B 相同 |
| `num_hidden_layers` | 32 | 8 ×（3 × GDN + 1 × full attention） |
| `layer_types` | `[L,L,L,F] × 8` | 24 层 GDN + 8 层全注意力 |
| `num_attention_heads` | 16 | 全注意力 |
| `num_key_value_heads` | 4 | 全注意力 |
| `head_dim` | **256** | ✅ KV-cache 量化存储的前提 |
| `linear_num_key_heads` | 16 | GDN，`kg = 16×128 = 2048` |
| `linear_key_head_dim` | 128 | 同上 |
| `linear_num_value_heads` | **32** | GDN，`vg = 32×128 = 4096` |
| `linear_value_head_dim` | 128 | 同上 |
| `linear_conv_kernel_dim` | 4 | GDN 卷积窗口 |
| `full_attention_interval` | 4 | |
| `mtp_num_hidden_layers` | 1 | MTP 投机解码 |
| `mtp_use_dedicated_embeddings` | `false` | MTP 复用主 embedding |
| `tie_word_embeddings` | `false` | 需要独立 `output_head` |
| `attention_bias` | `false` | |
| `attn_output_gate` | `true` | 所以有 `attention/gate` |
| `rope_parameters` | `mrope_interleaved=true`, `mrope_section=[11,11,10]`, `rope_theta=1e7`, `partial_rotary_factor=0.25` | RoPE 维度 = 0.25 × 256 = **64** |
| 特殊 token | `vision_start 248053`、`vision_end 248054`、`image 248056`、`video 248057`、`eos 248044` | vocab 248320 含这些 |

### 1.2 Vision 配置

| 字段 | 值 | 对适配的影响 |
|---|---|---|
| `hidden_size` | 1152 | ✅ 与 27B 相同 |
| `intermediate_size` | 4304 | ✅ 相同 |
| `depth` | 27 | 层数，不影响 (n,k) |
| `num_heads` | 16 | |
| `out_hidden_size` | **4096** | ❌ 27B 是 5120（=其文本 hidden）→ merger/fc2 需换 |
| `patch_size` / `temporal_patch_size` | 16 / 2 | patch_embedding 的 `k = 3×2×16×16 = 1536` |
| `spatial_merge_size` | 2 | merger 宽 = `2²×1152 = 4608` |
| `num_position_embeddings` | 2304 | |

### 1.3 推导量（用于后续所有表格）

```
kg        = linear_num_key_heads   × linear_key_head_dim   = 16 × 128 = 2048
vg        = linear_num_value_heads × linear_value_head_dim = 32 × 128 = 4096
q         = num_attention_heads    × head_dim              = 16 × 256 = 4096
kv        = num_key_value_heads    × head_dim              =  4 × 256 = 1024
hidden    = 4096      intermediate = 12288

# GDN 输入投影的"精确问题"六元组（见 §4.2）
qk_rows       = 2 × kg = 4096      # query + key 拼接
value_rows    = vg     = 4096
qkv_rows      = 2·kg + vg = 8192
z_rows        = vg     = 4096
value_z_rows  = 2 × vg = 8192
padded_k      = hidden = 4096
```

**⚠️ 重要更正**：`kg = 2048` 与 27B 相同（都是 16 × 128），但 **`vg` 不同**
（9B 是 32 头 → 4096；27B 的谓词 `value_z_rows == 12288` 反推 `vg = 6144` → 48 头）。
早期笔记把 GDN 头几何记为"跨模型不变量"，**对 value 侧是错的**，本文以谓词反推为准。

---

## 2. 根因：为什么现在必然 `unsupported shape`

`src/ops/linear/linear.cpp` 里**没有任何通用兜底**：每种量化格式都进各自的
`select_*_launch()` **闭表**（`ShapeEntry{n, k, select}`），未命中即
`throw std::invalid_argument("<fmt> linear: unsupported shape")`。
而且 `linear_workspace_capacity_bytes()` 在**规划期**就调用同一批选择器
→ **不是首次推理时才失败，而是加载/规划阶段就失败**。

q4/q5 现有表内只有 27B 几何（`k=5120`）与 vision 几何（`k=1152`）：

| 格式 | 现有 `(n,k)` |
|---|---|
| Q4 | `(1024,5120) (4096,5120) (5120,6144) (6144,5120) (7168,5120) (34816,5120) (131072,5120) (131072,2048) (3456,1152) (4304,1152)` |
| Q5 | `(1024,5120) (6144,5120) (7168,5120) (5120,6144) (5120,17408) (1152,1152) (1152,4304)` |
| Q6 | `(248320,5120) (248320,2048) (1152,1536)` |

→ 9B 需要的 `k = 4096 / 12288` **一个都没有**。

---

## 3. 准入分布在四层（这是本次勘察的核心发现）

适配一个模型不是改一处，而是要同时过**五层**（2026-09-16 修正，原记四层）。
**只看 `*_dispatch.cpp` 会严重低估工作量。**

| 层 | 位置 | 形态 | 改法 |
|---|---|---|---|
| **L1** | `src/ops/linear/<fmt>/` | `ShapeEntry` 闭表 + `select_q<F>_n<N>_k<K>()` 选择器 | 新增表项 + 选择器（**机械**） |
| **L2** | `src/ops/<op>/…/*_plan.cpp` **及同目录 `.cu`** | 硬编码"精确问题"元组 + 按列数路由 + **内核里烤死的几何** | 改谓词 + 加路由 + 参数化内核 |
| **L3** | `src/ops/wrapper/*.cpp` | 公共 op API 里另有一整套写死的 `n/k/rows` | 加分支（**易漏，必须逐文件过**） |
| **L4** | 跨层常量 | 工作区容量、stride 等 | 替换常量 |
| **L5** | `src/ops/weight_input.cpp` | **权重准备期**的独立闭表（`input_projection`、`prepare_gdn_gating_proj_weights`） | 加分支；漏了会在权重准备阶段就被拒 |

✅ 好消息：**`src/models/`、`src/core/`、`src/runtime/`、`src/serve/` 均无硬编码模型几何**
（已用 Grep 逐目录核实）。约束全部收敛在 `src/ops/`，符合 `AGENTS.md` 的分层意图。

---

## 4. Gap 清单

### 4.0 前置：先确定每种投影用哪个格式

格式由 `tools/convert/official_recipes.py` 决定，规则如下（`_dense_groupwise` + `_optional`）：

| 组件 | 格式 | 依据 |
|---|---|---|
| `text/token_embedding`、`text/output_head` | **Q6** | `_assign(..., vocabulary)`，27B 也是 Q6 |
| `…/attention/query`、`…/attention/key` | **Q4** | `_dense_groupwise` 的 Q4 名单 |
| `…/gdn/query`、`…/gdn/key` | **Q4** | 同上 |
| `…/mlp/gate`、`…/mlp/up` | **Q4** | 同上 |
| 其余 `text/layers/**` 投影（`attention/gate`、`/value`、`/output`、`gdn/value`、`gdn/z`、`gdn/output`、`mlp/down`） | **Q5** | 兜底 |
| `…/gdn/a_projection`、`…/gdn/b_projection` | `separate`（fp32） | 显式分离 |
| **`vision/merger/**`** | **Q8** | `_optional` |
| `vision/patch_embedding` | Q6 | `_optional` |
| `vision/**/attention/{query,key,value}`、`/mlp/fc1` | Q4 | `_optional` |
| 其余 `vision/**` | Q5 | `_optional` |
| **`mtp/**`（MTP 层）** | **Q8** | `_optional`（`mtp`/`dflash`/`dflash2` 统一 Q8） |

> ⚠️ 关键推论：**MTP 与 vision merger 走 Q8**。而 q8 在 sm_89 上是被部分剔除过的路径
> （7 个内核因静态 `__shared__` 超 48 KB 被门控掉）。因此
> **"纯文本 + 不带 MTP + 不带 vision"是最短的可用路径**，全程只需 Q4/Q5/Q6。

### 4.1 阶段划分

| 阶段 | 组件 | 需要的格式 | 是否可独立交付 |
|---|---|---|---|
| **S1（目标：先跑起来）** | `--components text`，不启用投机解码 | Q4 / Q5 / Q6 | ✅ 是 |
| S2 | + MTP 投机解码 | 追加 Q8（hidden 4096 全系） | ✅ 是 |
| S3 | + vision | 追加 Q8 `(4096,4608)` 一条 | ✅ 是 |

### 4.2 L1：`linear` 闭表新增条目（S1）

**Q4** — `src/ops/linear/q4/q4_dispatch.cpp:12-22`（表）+ `q4/q4_shapes.h`

| 新增 `(n,k)` | 服务的投影 | 来源 |
|---|---|---|
| `(4096,4096)` | `attention/query`；亦覆盖 `gdn` 的 `group(query,key)` | q=4096, h=4096 |
| `(1024,4096)` | `attention/key` | kv=1024 |
| `(2048,4096)` | `gdn/query`、`gdn/key`（各自） | kg=2048 |
| `(5120,4096)` | `group(attention/query, key)` 融合权重 | 4096+1024 |
| `(12288,4096)` | `mlp/gate`、`mlp/up`（各自） | intermediate |
| `(24576,4096)` | `group(mlp/gate, up)` 融合权重；亦被 `linear_swiglu` 的 `Materialized` 路由以 `linear()` 调用 | 2×12288 |
| `(131072,4096)` | ⚠️ **仅当启用 `--proposal`**（proposal head 行数） | `--proposal-rows` 默认 131072 |

**Q5** — `src/ops/linear/q5/q5_dispatch.cpp:12-18`（表）+ `q5/q5_shapes.h`

| 新增 `(n,k)` | 服务的投影 |
|---|---|
| `(4096,4096)` | `attention/gate`、`attention/output`、`gdn/value`、`gdn/z`、`gdn/output` |
| `(1024,4096)` | `attention/value` |
| `(5120,4096)` | `group(attention/gate, value)` 融合权重 |
| `(8192,4096)` | `group(gdn/value, z)` 融合权重（2×4096） |
| `(4096,12288)` | `mlp/down` |

**Q6** — `src/ops/linear/q6/q6_dispatch.cpp:12-16`（表）+ `q6/q6_shapes.h`

| 新增 `(n,k)` | 服务的投影 |
|---|---|
| `(248320,4096)` | `token_embedding`、`output_head` |

> **S1 合计：Q4 六条（不计 proposal）、Q5 五条、Q6 一条 = 12 条闭表项。**
> 阶段 S2/S3 还要追加 Q8 的 `(4096,4096) (1024,4096) (6144,4096) (12288,4096) (4096,12288)`
> 与 `(4096,4608)`。

### 4.3 L2：精确问题算子的 plan（S1）

| 算子 | 位置 | 现有（27B）元组 | 9B 需要的元组 |
|---|---|---|---|
| `attn_input_proj` q4_q5 | `src/ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.cpp:10-13` | `input_rows 5120, query_rows 6144, kv_rows 1024, padded_k 5120` | `input_rows 4096, query_rows 4096, kv_rows 1024, padded_k 4096` |
| `gdn_input_proj` q4_q5 | `src/ops/gdn_input_proj/q4_q5/q4_q5_gdn_input_plan.cpp:41-44` | `input_rows 5120, qk_rows 4096, value_z_rows 12288, qkv_rows 10240, z_rows 6144, padded_k 5120` | `input_rows 4096, qk_rows 4096, value_z_rows 8192, qkv_rows 8192, z_rows 4096, padded_k 4096` |
| `linear_add` **q4** | `src/ops/linear_add/q4/q4_linear_add.cu:73`（谓词）、`:59`（`Q4LinearGeometry<5120,6144>`）、`:33`（`GemvR1W8` 里 `6144/64`）、`:43,47`（epilogue 硬编码 stride `5120`） | `rows 5120, k 6144` —— **编译期** | `rows 4096, k 4096` |
| `linear_add` **q5** | `src/ops/linear_add/q5/q5_linear_add_plan.cpp:35-38`（`kSupports`）、`41-58`（两套路由 `kK6144Routes`/`kK17408Routes`） | `(5120,6144,6144)` 与 `(5120,17408,17408)`；路由按 `k == 6144 ? A : B` **二元硬切** | 追加 `(4096,4096,4096)` 与 `(4096,12288,12288)`，并把二元硬切改为**按 k 索引的路由表** |
| `linear_swiglu` **q4** | `src/ops/linear_swiglu/q4/q4_linear_swiglu_plan.cpp:33`（`kShape`）、`:60-68`（谓词） | `{gate_up_rows 34816, output_rows 17408, k 5120, padded_k 5120}` | `{gate_up_rows 24576, output_rows 12288, k 4096, padded_k 4096}` |

> ⚠️ `q4_linear_add` 是**唯一把 (n,k) 真正烤进内核**的算子族（模板参数 + stride 常量）。
> 两条改法：(a) 新增 `Q4LinearGeometry<4096,4096>` 实例并把 stride 参数化；
> (b) 让 stride/行数走运行时入参。**(a) 风险更低**（不影响 5090 的既有 codegen）。

### 4.4 L3：公共 op API 里的硬编码（S1，**最易漏**）

| 文件:行 | 现有条件 | 9B 需要的改动 |
|---|---|---|
| `src/ops/wrapper/attn_input_proj.cpp:45` | `weight.k != 5120` | 允许 `4096` |
| `src/ops/wrapper/attn_input_proj.cpp:185` | `parent_rows != 14336 \|\| input_rows != 5120`（27B 分支） | 新增 9B 分支：`parent_rows == 24576 && input_rows == 4096` |
| `src/ops/wrapper/attn_input_proj.cpp:207` | `9216 \|\| 2048`（35B-A3B 分支） | 不动 |
| `src/ops/wrapper/attn_input_proj.cpp:265` | `hidden != 2048 && hidden != 5120` | 追加 `4096` |
| `src/ops/wrapper/gdn_input_proj.cpp:264` | `weight.k != 5120` | 允许 `4096` |
| `src/ops/wrapper/gdn_input_proj.cpp:276` | `weight.k != 2048`（35B-A3B） | 不动 |
| `src/ops/wrapper/gdn_input_proj.cpp:759` | q8：`parent_rows == 12288 && input_rows == 2048` | S2/S3 再处理 |
| `src/ops/wrapper/gdn_input_proj.cpp:783,865` | q4_q5：`query_rows == 2048 && key_rows == 2048 && value_rows == 6144` | 追加：`query 2048 && key 2048 && value 4096` **且 `hidden == 4096`**（9B 的 value 宽度恰好与 35B-A3B 的 `value_rows == 4096` 相同，**必须用 hidden 区分**） |
| `src/ops/wrapper/linear_add.cpp:194` | `(n 5120 && k 17408) \|\| (n 5120 && k 6144)` | 追加 `(4096,12288)`、`(4096,4096)` |
| `src/ops/wrapper/linear_swiglu.cpp:82-85` | `gate_up 34816 && k 5120 …`（另有 35B-A3B 的 q8 分支） | 追加 9B 的 q4 分支 `gate_up 24576 && k 4096` |
| ~~`src/ops/wrapper/dynamic_grouped_conv.cpp`~~ | `kHidden = 5120`、`kGroups = 320`（= 5120/16）、`:206` `input_rows != 4096 && != 17408` | ✅ **S1 不涉及**：该文件是 **DFlash2 专属**（`kGroups = h / conv_group_size = 5120/16 = 320`），只在 `dflash2` 组件启用时用到。9B 的 GDN 卷积走 `gdn_input_proj_conv` 的 `ProjectionEpilogueFused`/`Materialized` 调度，不在此处 |
| `src/ops/wrapper/gdn_gating_proj.cpp:52,56` | `parent.n == 96 && k == 5120`（27B）、`n == 64 && k == 2048`（35B-A3B） | 追加 9B：`parent.n == 64 && parent.k == 4096`（a/b 各 32 行 → 64，k=hidden 4096） |
| `src/ops/wrapper/linear_pair.cpp:96` | `first_weight.k != 5120 && != 2048` | 追加 `4096` |
| `src/ops/wrapper/linear_topk/q4.cu:43` | `Q4LinearGeometry<131072, kLinearTopKHidden>` | 仅当启用 proposal head 时核对 |

### 4.5 L4：跨层常量

| 位置 | 现有 | 9B | 说明 |
|---|---|---|---|
| ~~`src/ops/dynamic_grouped_conv/bf16/bf16_dynamic_grouped_conv_prepare_plan.cpp:18`~~ | `capacity = 1280 * tokens * split_k * 4` | — | ✅ **S1 不涉及**：与 §4.4 同源，属 DFlash2 组件。若将来启用 `dflash2`，需连同 `kHidden`（→5120/4096）、`kGroups`（= `h / conv_group_size`）与 `*_partial.cu`/`*_reduce.cu` 的 stride 一起重算 |

### 4.6 ✅ 无需改动的部分（已核实）

| 项 | 依据 |
|---|---|
| **L5 `src/ops/weight_input.cpp`** | ✅ **已完成（无需改动）**：`input_projection()` L126–138 已列 Qwen3.5 Small 9B（hidden 4096）的 `(2048,4096)/(4096,4096)` 形状，L174–175 的 `require` 也已接受 `{4096, 4096}`；`prepare_gdn_gating_proj_weights()` L202 的 `(32,4096)` 分支经核实**本已存在** |
| **KV-cache 存储** | `src/core/paged_kv_storage.h:12,83,86,91,94` 要求 `head_dim == kD256KVCacheHeadDim = 256`；9B 的 `head_dim` 正是 256 → `Int8Group64` / `Fp8E4M3Row256` / `BFloat16` 全部可用 |
| `src/models/`、`src/core/`、`src/runtime/`、`src/serve/` | Grep 证实**零**硬编码模型几何 |
| vision 的其它投影 | `patch_embedding (1152,1536)` Q6、`attention q/k/v (3456,1152)` Q4、`attention/output (1152,1152)` Q5、`mlp/fc1 (4304,1152)` Q4、`mlp/fc2 (1152,4304)` Q5 —— **全部已在表内** |
| GDN 的 `kg = 2048` | 与 27B 一致（都是 16 × 128） |

---

## 5. 新增选择器清单（S1）

对 §4.2 的每一条 `(n,k)`，需要三处改动。**推荐一律复用通用 launcher**
（`launch_q4_gemv_r4_w1_direct` / `launch_q4_simt_r8_c4` / `launch_q4_simt_r8_c8` /
`launch_q4_mma_r64_c128`；`launch_q5_simt_r8_c4/c8` / `launch_q5_mma_r64_c64/c128`），
**不要**照抄 `.cu` 形状里的专用 tile：

> 理由：专用 tile（`launch_q4_ksplit<N,K,T>` / `launch_q4_mma<Schedule>`）会把 n、k 烤进模板，
> 且在 sm_89 上有**静态 `__shared__` 超 48 KB** 的风险（q8 已被此坑掉 7 个内核）。
> 通用 launcher 已由 `ninfer_linear_q4_a16_test` 等测试在真机验证通过（见 §7 G2）。

### 5.1 文件模板

`src/ops/linear/q4/shapes/n4096_k4096.cpp`：

```cpp
#include "ops/linear/q4/q4_shapes.h"

namespace ninfer::ops::detail {

Q4Launch select_q4_n4096_k4096(std::int32_t tokens) {
    if (tokens == 1) return launch_q4_gemv_r4_w1_direct;   // ← 不是 r1_q8_direct！见下方陷阱说明
    if (tokens <= 4) return launch_q4_simt_r8_c4;
    if (tokens <= 16) return launch_q4_simt_r8_c8;
    return launch_q4_mma_r64_c128;
}

} // namespace ninfer::ops::detail
```

🚨 **陷阱：三个"看起来通用、实则把形状烤死"的 launcher**（实测源码，`§15` 之后补充发现）：

| launcher | 真实约束 | 结论 |
|---|---|---|
| `launch_q4_gemv_r1_q8_direct` | `Q4RowSplitGemvSchedule<..., 80, 1>`，`kStaticGroupsPerRow = 80` = **5120/64** | ❌ 只对 `k == 5120` 正确；k=4096 时 `groups_per_row = 80` 会**越界读** |
| `launch_q4_gemv_r4_w1_direct` | `Q4RowSplitGemvSchedule<..., 0, 1>`，`kStaticGroupsPerRow = 0` → 走 `k / kGroupK` 运行时分支 | ✅ 真通用（已被 k=2048 与 k=5120 两种形状使用） |
| `launch_q5_gemv_r16_s2_x` | 体内 `constexpr int kK = 5120;`，且 `w.n` 只认 `{6144, 7168}`，否则 `throw` | ❌ **绝不可用**；q5 的新形状一律回落到 `launch_q5_simt_r8_c4/c8` |

判据（可复用）：`.cuh` 的 schedule 模板末位若是 `StaticGroupsPerRow`，**`> 0` 即把 k 烤死**（`q4_rowsplit_gemv.cuh:454`）；
`= 0` 才是按运行时 k 推导。q5 的 GEMV 则把 K 做成了模板参数（`q5_rowsplit_gemv_launch_kernel<Rows, K, ...>`），**没有**运行时 K 的变体。


### 5.2 改动点清单

| # | 文件 | 动作 |
|---|---|---|
| 1 | `src/ops/linear/q4/shapes/n{4096,1024,2048,5120,12288,24576}_k4096.cpp` | 新建（6 个，`.cpp`） |
| 2 | `src/ops/linear/q4/q4_shapes.h` | 6 行声明 |
| 3 | `src/ops/linear/q4/q4_dispatch.cpp:12-22` | 6 行 `ShapeEntry` |
| 4 | `src/ops/linear/q5/shapes/n{4096,1024,5120,8192}_k4096.cpp` + `n4096_k12288.cpp` | 新建（5 个） |
| 5 | `src/ops/linear/q5/q5_shapes.h` | 5 行声明 |
| 6 | `src/ops/linear/q5/q5_dispatch.cpp:12-18` | 5 行 `ShapeEntry` |
| 7 | `src/ops/linear/q6/shapes/n248320_k4096.cpp` | 新建（1 个） |
| 8 | `src/ops/linear/q6/q6_shapes.h` | 1 行声明 |
| 9 | `src/ops/linear/q6/q6_dispatch.cpp:12-16` | 1 行 `ShapeEntry` |
| 10 | `src/CMakeLists.txt:227-236`（q4 段）、`:237-244`（q5 段）、`:288-290`（q6 段） | ⚠️ **显式源文件列表，无 GLOB** → 每个新文件必须手工入列，漏了会链接失败 |

> ⚠️ 若 Q4 表新增 `(24576,4096)`，注意 `q4_linear_swiglu` 的 `Materialized` 路由会以
> `linear(x, w, gate_up)` 复用 `linear` 的闭表 —— 两处必须同时有该条目。

### 5.3 ✅ 执行记录（第 1 步已完成）

**改动清单（12 新建 + 6 修改）**

| 类别 | 文件 | 动作 |
|---|---|---|
| Q4 | `src/ops/linear/q4/shapes/n{1024,2048,4096,5120,12288,24576}_k4096.cpp` | 新建 6 个（全部 `.cpp`，纯通用 launcher） |
| Q4 | `src/ops/linear/q4/q4_shapes.h` / `q4_dispatch.cpp` | 6 行声明 + 6 行 `ShapeEntry` |
| Q5 | `src/ops/linear/q5/shapes/n{1024,4096,5120,8192}_k4096.cpp`、`n4096_k12288.cpp` | 新建 5 个 |
| Q5 | `src/ops/linear/q5/q5_shapes.h` / `q5_dispatch.cpp` | 5 行声明 + 5 行 `ShapeEntry` |
| Q6 | `src/ops/linear/q6/shapes/n248320_k4096.cpp` | 新建 1 个 |
| Q6 | `src/ops/linear/q6/q6_shapes.h` / `q6_dispatch.cpp` | 1 行声明 + 1 行 `ShapeEntry` |
| 构建 | `src/CMakeLists.txt` | 12 行源文件入列 |
| 测试 | `tests/ops/linear/test_{q4,q5,q6}_a16.cpp` | 各追加新几何的 `run_shape` 用例 + `verify_workspace_envelopes` |

**实际选用的 ladder**（与 §5.1 的模板有一处修正）

```cpp
// Q4（k=4096）：不使用 launch_q4_gemv_r1_q8_direct
if (tokens == 1) return launch_q4_gemv_r4_w1_direct;   // 唯一真通用的 Q4 GEMV
if (tokens <= 4) return launch_q4_simt_r8_c4;
if (tokens <= 16) return launch_q4_simt_r8_c8;
return launch_q4_mma_r64_c128;

// Q5（k=4096/12288）：没有可用的通用 GEMV，t=1 也走 SIMT
if (tokens <= 4) return launch_q5_simt_r8_c4;
if (tokens <= 16) return launch_q5_simt_r8_c8;
return launch_q5_mma_r64_c128;

// Q6（n=248320, k=4096）：镜像同 n 的 k=5120 兄弟
if (tokens <= 4) return launch_q6_simt_r8_c4;
if (tokens <= 5/6/7) return launch_q6_simt_r8_c5/c6/c7;
if (tokens <= 16/24/32/48) return launch_q6_mma_r64_c{16,24,32,48}_k128;
return launch_q6_mma_r64_c128;
```

**验证结果（G1/G2 通过）**

| 门禁 | 命令 | 结果 |
|---|---|---|
| G1 | `msvcbuild.py configure 89 -DBUILD_TESTING=ON` + 构建 apps 与 7 个测试目标 | **rc=0，零 FAILED**；配置行 `architectures: 89 (NVFP4 kernels: OFF; PDL: OFF; large static __shared__: OFF)` |
| G2 | `ctest -R "ninfer_(device\|linear_q4_a16\|linear_q5_a16\|linear_q6_a16\|linear_add_q4_a16\|linear_add_q5_a16\|linear_swiglu_q4_a16)_test"` | **7/7 Passed**（26.4 s），且三个线性测试 stdout 均为 `OK ... Linear`（非 SKIP） |
| G6（提前做） | 同上 | 12 个新 `(n,k)` 已全部纳入 `run_shape` + fp64 oracle 校验，**不是"未验证的内核"** |
| — | `ninfer.exe <dummy>.bin --prompt hi` | rc=1，`starting engine` → `error: NInfer accepts only .ninfer artifacts`（应用级拒绝，无回归） |

**顺带确认的两项**

- 闭表只被 `src/ops/linear/linear.cpp` 消费（Grep 全仓核实），改表不会波及其它层。
- `linear_workspace_capacity_bytes()` 对 Q4/Q5/Q6 返回 0 —— 新几何不需要额外工作区。

⚠️ **重建构建树 / clean 之后必须重跑 `msvcbuild.py deploycrt 89`**，否则复现 §15.3 的 `MSVCP140.dll` 崩溃。

---

## 6. 权重下载与 recipe

### 6.1 下载

| 项 | 值 |
|---|---|
| 仓库 | `Qwen/Qwen3.5-9B`（后训练版；若要 base 则 `Qwen/Qwen3.5-9B-Base`） |
| 体积 | BF16 ≈ **18 GB**（磁盘需预留 ≥ 40 GB：源 + 产物 + 临时） |
| 工具 | `huggingface-cli download Qwen/Qwen3.5-9B --local-dir D:\models\Qwen3.5-9B` |

> ⚠️ 本机 HF 网络此前有超时史（见 `PilotTTS` 经验）。必要时先设镜像端点，
> 或按需只取 `*.safetensors` 与 `config.json` / `tokenizer.json`。
> 磁盘 `D:\` 需 ≥ 40 GB 可用。

### 6.2 recipe

`tools/convert/official_recipes.py` 追加（与 `qwen3_6_27b` 同构，因为 9B 是 dense）：

```python
def qwen3_5_9b(model, recipe, sources):
    _dense_groupwise(model, recipe, Q6)
```

并入 `RECIPES` 字典：

```python
RECIPES = {
    ...
    "qwen3_5_9b": qwen3_5_9b,
}
```

> `_dense_groupwise` 会先校验 `num_experts not in model.config`。9B 无 `num_experts`、
> `mlp_only_layers` 为空 → 密集合规。

### 6.3 转换命令

```bash
# 转换器要求 --device cuda（量化在 GPU 上做）
python -m tools.convert \
  --model D:/models/Qwen3.5-9B \
  --recipe qwen3_5_9b \
  --components text \
  --name qwen3.5-9b-q456 \
  --out D:/models/qwen35-9b.ninfer
```

`--components text` 是默认值，**刻意不含 `vision` / `mtp`**（见 §4.0 的 Q8 推论）。

### 6.4 显存预算

| 项 | 估算 | 依据 |
|---|---|---|
| 权重（Q4/Q5/Q6 混合，文本） | **≈ 5.5–6.0 GB** | q4_g64≈4.5 bpw、q5_g64≈5.5、q6_g64≈6.5；embed+head 2×248320×4096 占 6.5 bpw |
| GDN 递归态（24 层，f32） | **≈ 0.2 GB** | 独立实现（llama.cpp 在同类模型上实测 192 MiB S + 9 MiB R） |
| KV cache（8 层全注意力，Int8 K / FP16 V） | **≈ 25 KB/token** → 4K ctx ≈ **0.1 GB** | 每层每 token：K 4×256×1B + V 4×256×2B = 3072 B；×8 层 |
| 计算图/临时缓冲 | ≈ 0.2–0.4 GB | 按 27B 的比例外推 |
| **合计（4K ctx）** | **≈ 6.1–6.7 GB** | 8188 MiB 可容纳 ✅ |
| 可用上下文上限 | KV 每 token 仅 25 KB → **32K ctx 也只多 0.7 GB** | 实际上限由权重+递归态决定 |

> 建议起步：`--kv-dtype int8 --max-context 4096`，稳定后再逐级放大。
> ⚠️ 以上权重体积为**估算**，最终以转换产物的实际文件大小为准（见 §7 G3）。

---

## 7. 验证计划（逐阶段门禁）

| 门禁 | 内容 | 通过标准 | 状态 |
|---|---|---|---|
| **G1** | `msvcbuild.py configure 89 -DBUILD_TESTING=ON` + 全量构建 | rc=0，零 `FAILED` | ✅ 第 1 步后已过（§5.3） |
| **G2** | **回归**：测试在真机通过 <br>`ninfer_{device,linear_q4_a16,linear_q5_a16,linear_q6_a16,linear_add_q4_a16,linear_add_q5_a16,linear_swiglu_q4_a16}_test` | N/N `Passed`（防止改 L2/L3 时打坏 27B 路径） | ✅ 7/7（§5.3） |
| **G3** | 转换产出 `.ninfer` | 文件存在，大小落在 §6.4 估算区间 | ⬜ |
| **G4** | `msvcbuild.py env apps/ninfer.exe <model>.ninfer --prompt "你好"` | 正常出 token，**无 `unsupported shape`** | ⬜ |
| **G5** | `nvidia-smi` 峰值显存 | < 8188 MiB，留 ≥ 300 MiB 余量 | ⬜ |
| **G6**（推荐） | 为新增 `(n,k)` 补 op 测试条目（`tests/ops/linear/`） | 用 `quantized_weight.h` 确定性权重 + 独立 oracle 校验；**每个新几何都要有**，否则等于"未验证的内核" | ✅ **L1 的 12 个已全部覆盖**（§5.3）；L2/L3 落地后需再补 |
| **G7** | 27B 路径不回归 | 若能拿到 27B 产物，跑一次同样命令 | ⬜ |

> ⚠️ 每次**重建构建树或 clean 之后**必须重跑 `msvcbuild.py deploycrt 89`，
> 否则会复现 §15.3 那个 `MSVCP140.dll` 崩溃（本机 System32 VC 运行时是错版）。

---

## 8. 风险与回滚

| 风险 | 等级 | 缓解 |
|---|---|---|
| 新几何的专用 tile 在 sm_89 上静态 `__shared__` 超 48 KB | ⚠️ 中 | 一律用通用 launcher；必要时按 `NINFER_ENABLE_LARGE_STATIC_SMEM` 模式加桩 |
| 新 `(n,k)` 未经真机数值验证 | ❌ 高 | G6：每个新几何补 oracle 测试 |
| `q4_linear_add` 改动触及 5090 既有 codegen | ⚠️ 中 | 只**新增** `Q4LinearGeometry<4096,4096>`，不改 `<5120,6144>` |
| L3 wrapper 分支判断歧义（9B 的 `value_rows 4096` 与 35B-A3B 撞车） | ⚠️ 中 | 分支条件必须**同时**约束 `hidden`，见 §4.4 `gdn_input_proj.cpp:783` |
| `dynamic_grouped_conv` 的 stride 与容量常量不一致 | ⚠️ 中 | 先核对 partial/reduce 内核的索引方式，再定改法 |
| Q8 在 sm_89 的可用性（S2/S3） | ⚠️ 中 | S1 完全绕开；S2 前先用 `ninfer_linear_q8_a16_test` 在真机验一次 |
| 下载 18 GB + 转换耗时 | ⚠️ 低 | 预留磁盘与时间；转换可断点重试 |

**回滚策略**：所有改动均为**新增**（新文件 + 新表项 + 新分支），不删除、不重写现有条目
→ 出现问题时可按文件粒度回退，27B / 35B-A3B 路径不受影响。

---

## 9. 执行顺序（S1）

| 步 | 内容 | 产出 | 状态 |
|---|---|---|---|
| 1 | L1：12 个新形状选择器 + 3 张闭表 + `src/CMakeLists.txt` 入列 | 构建通过 | ✅ **已完成**（§5.3） |
| 2 | L2：`attn_input_proj` / `gdn_input_proj` / `linear_swiglu` q4 的谓词扩展 | 规划期不再拒绝 9B 元组 | ✅ **已完成**（2026-09-16，交付小结 §L2） |
| 3 | L2：`linear_add` q4（几何实例化 + stride 参数化）与 q5（`kSupports` + 路由表按 k 索引） | | ✅ **q5 已完成**（2026-09-16：`(4096,4096)`/`(4096,12288)` + 四路 k 索引路由 + gemv/simt 实例化，测试全过）；q4 经核实 **S1 不需要**——9B 的 Q4 只用于输入侧投影（input proj / swiglu），无 residual-add 路由 |
| 4 | L3：`src/ops/wrapper/` 逐文件加 9B 分支（按 §4.4 表逐行过） | | ✅ **gating/linear_add 已完成**（2026-09-16）；attn/gdn input proj、linear_swiglu 随 L2 完成；**遗留 `linear_pair.cpp:96`**（9B 是否用到 pair 待确认，proposal/dflash 关闭时不触发） |
| 5 | L4：~~`bf16_dynamic_grouped_conv_prepare` 的常量~~ → 已确认属 DFlash2，**S1 无需改动** | | ✅ 无需改动 |
| 6 | **G1 + G2**：构建 + 测试回归 | 无回归 | ✅ 12 套件 11/12（唯一 FAIL 为 27B T=4097 fp32 既有边界，见 L2 交付小结） |
| 7 | 下载权重 + 补 recipe + 转换（§6） | `qwen35-9b.ninfer` | ✅ **已完成**（2026-09-16 晚）：hf-mirror 下载 18.4GB（分片 3 curl 断点续传）；`qwen3_5_9b` recipe；CPU 转换 79s/387 张量 → `D:\deps\models\qwen35-9b-final.ninfer`。转换器需 Windows 兼容补丁（`file_io.py` pread/pwrite 降级 + 4 处 `os.open` 加 **O_BINARY**）与手写 generation_config.json、chat_template 双资源对齐 |
| 8 | **G4 + G5**：真机推理 + 显存核对 | ✅ 目标达成 | ✅ **冒烟成功**（2026-09-16 晚）：decode **38.2 tok/s**、prefill 201 tok/s、权重 5.28 GiB、planned total 5.52 GiB。运行时补 4 处：`weight_input.cpp` 配对回退加 32 行、`wrapper/gdn_gating_proj.cpp` 配对重载几何化、`startup.cpp:804` 架构门放行 89、chat_template 用引擎已知 digest（LF 去尾）资源覆盖。~~kv_cache 9B~~ 已证伪无需改 |
| 9 | G6：补新几何的 op 测试 | 数值正确性闭环 | ✅ gating（7 个 T 全路由 + norm Composed + replay）、linear_add q5 两个新形状全路由已覆盖 |
| 10 | 文档与提交（含 §16 里 H/I 两项待办） | | ⬜ |

> 建议 **1–4 步先做完再下载权重**（下载/转换是长尾，且与代码改动无耦合）。
> 但第 6 步的门禁必须在下载之前跑绿。
> ⚠️ 第 2–4 步每做完一步都要**重跑 G2**（L2/L3 的改动最容易打坏 27B 的既有分支）。

---

## 10. 尚待确认的开放项

| # | 项 | 确认方式 |
|---|---|---|
| ~~A~~ | ~~`dynamic_grouped_conv` 在 9B 下的 `input_rows`~~ | ✅ **已确认**：DFlash2 专属，S1 不涉及（§4.4 / §4.5） |
| ~~B~~ | ~~`bf16_..._prepare` 的 partial/reduce 是否按 `C` 索引~~ | ✅ **已确认**：同上，属 DFlash2 |
| C | 转换产物是否真的需要 `(131072,4096)`（proposal 默认关，MTP 才用） | 看 `--proposal` 是否被 CLI/引擎默认启用 |
| D | 9B 是否启用 `dflash`/`dflash2`（都会引入 Q8 且形状更多） | 查 9B 的 MTP/draft 配置；S1 一律关 |
| E | 引擎对 `layer_types` 混合布局的现有支持是否已覆盖 3:1 比 | 已有 27B 同构（`full_attention_interval=4`）→ 预计无需改 |
| F | GDN 递归态（f32）在 sm_89 上的实际显存占用 | 首次运行后用 `nvidia-smi` 核对 §6.4 的 0.2 GB 估算 |
