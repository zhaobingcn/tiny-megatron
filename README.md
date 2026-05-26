ni-Megatron PyTorch Learning Plan

目标：仿照 Megatron-LM / Megatron Core 的设计，用 PyTorch 写一个自学版训练框架。范围从单卡 GPT 训练循环开始，逐步加入配置系统、数据管线、分布式数据并行、张量并行、流水并行、混合精度、checkpoint 和测试。

## 0. 参考地图

先把 Megatron 拆成四层来学，不一开始照搬全部复杂度。

| 层级 | Megatron 参考 | 自学框架目标 |
| --- | --- | --- |
| 训练入口 | `pretrain_gpt.py`, `megatron/training/training.py` | `train.py` 负责解析配置、初始化、循环、保存 |
| 模型组件 | `megatron/core/models`, `megatron/core/transformer` | 手写 GPT: embedding, attention, MLP, block, LM head |
| 并行状态 | `megatron/core/parallel_state.py` | 管理 DP/TP/PP process group |
| 训练调度 | `megatron/core/pipeline_parallel/schedules.py` | 从普通 forward/backward 到 microbatch 和 pipeline schedule |
| 数据 | `megatron/core/datasets/gpt_dataset.py` | token dataset + packed sequence + causal labels |
| checkpoint | `megatron/training/checkpointing.py`, `megatron/core/dist_checkpointing` | 保存模型、优化器、scheduler、rng、step |
| 测试 | `tests/unit_tests`, `tests/functional_tests` | 单元测试覆盖 shape、loss、并行一致性、小步训练 |

## 1. 8 周主线

### Week 1: 单卡最小 GPT 训练闭环

交付物：
- `minim/core/model.py`: GPTConfig, CausalSelfAttention, MLP, TransformerBlock, GPTModel。
- `minim/data/tiny_dataset.py`: 从纯文本构造 token 序列，输出 `input_ids` 和 `labels`。
- `minim/train.py`: 单卡训练 100 step，打印 loss，保存 checkpoint。
- `tests/test_model_shapes.py`: 验证 logits shape、loss 可 backward。

学习重点：
- Megatron 的 `model_provider()` 思想：训练框架不直接硬编码模型，而是接收一个模型构造函数。
- 训练 loop 的最小边界：data iterator -> forward step -> loss -> backward -> optimizer step -> log。

完成标准：
- CPU 或单 GPU 能跑通 tiny corpus。
- loss 不是 NaN，checkpoint 能恢复继续训练。

### Week 2: 配置系统和训练状态

交付物：
- `minim/config.py`: dataclass 配置，包含 model/data/train/optimizer/checkpoint。
- `configs/gpt_tiny.yaml`: 最小配置文件。
- `minim/state.py`: TrainState(step, consumed_tokens, rng_state)。

学习重点：
- 参考 Megatron `arguments.py` 和 `training_config.py`，但自学版用 dataclass + yaml，避免过早引入复杂 CLI。
- 所有训练行为都来自配置，代码中只保留合理默认值。

完成标准：
- 改 yaml 即可调整层数、hidden size、batch size、lr、保存间隔。

### Week 3: 数据管线和 microbatch

交付物：
- `minim/data/text_dataset.py`: 支持 `.txt` 文件，固定 seq_len 切块。
- `minim/schedules.py`: `forward_backward_step(forward_step, data_iter, model, num_microbatches)`。
- gradient accumulation 支持 global batch = micro batch * accumulation steps。

学习重点：
- Megatron 把 forward step 写成回调，schedule 控制 microbatch、backward 和 loss 收集。
- 先不做 pipeline，只复刻“调度器拥有训练节奏”这一点。

完成标准：
- accumulation=1 和 accumulation=N 都能训练。
- 有测试验证 accumulation 后 optimizer step 次数正确。

### Week 4: torch.distributed + DDP

交付物：
- `minim/distributed.py`: 初始化 process group、rank/world_size/local_rank。
- `minim/parallel_state.py`: 先只管理 data parallel group。
- `torchrun --nproc-per-node 2 minim/train.py --config ...` 能跑。

学习重点：
- 参考 `parallel_state.py` 的全局并行状态思想，但只实现最小 DP。
- 正确处理 sampler seed、rank 0 logging、rank 0 checkpoint。

完成标准：
- 1 GPU 和 2 GPU 的 loss 曲线趋势相近。
- checkpoint 只由 rank 0 写出，所有 rank 能同步退出。

### Week 5: 混合精度、梯度裁剪和 scheduler

交付物：
- AMP bf16/fp16 训练开关。
- gradient clipping。
- cosine 或 linear warmup scheduler。
- optimizer state 恢复测试。

学习重点：
- Megatron 的训练 loop 里 optimizer、scheduler、grad finalize 是独立阶段；自学版也保持这个结构。

完成标准：
- fp32/bf16 均能跑。
- 恢复 checkpoint 后 lr、step、loss 继续衔接。

### Week 6: 张量并行 TP v0

交付物：
- `minim/tensor_parallel/layers.py`: ColumnParallelLinear, RowParallelLinear。
- `minim/parallel_state.py`: tensor parallel group。
- 在 attention/MLP 中替换部分 Linear。

学习重点：
- Megatron 的 TP 本质是把矩阵乘拆到多个 rank，并在必要处 all-gather/reduce-scatter/all-reduce。
- 先实现线性层级别 TP，不急着优化通信重叠。

完成标准：
- TP=1 和 TP=2 在同 seed、小模型下输出/梯度可对齐到合理误差。

### Week 7: Pipeline Parallel v0

交付物：
- 按 layer 切分模型到不同 rank。
- 实现 naive pipeline：先不做 1F1B，只做 microbatch 顺序 send/recv。
- 最后 stage 计算 loss，反向传播梯度。

学习重点：
- 先理解 stage 边界、激活传输、loss 只在最后一段计算。
- 再读 Megatron 的 1F1B schedule，不急着复刻全部。

完成标准：
- PP=2 可以完成几步训练。
- 有文档画出每个 microbatch 的 forward/backward 时间线。

### Week 8: 工程化收尾

交付物：
- README: 架构图、运行命令、当前支持矩阵。
- smoke tests: 单卡、DDP、TP tiny case。
- profiling: 每 step tokens/sec、显存、耗时分解。

学习重点：
- 一个训练框架不只是模型代码，还包括可复现、可恢复、可验证、可观测。

完成标准：
- 新用户能按 README 从零跑通 tiny GPT。
- 每个核心模块都有至少一个最小测试。

## 2. 每天执行节奏

每天 90-120 分钟，按固定顺序推进：

1. 读 20 分钟 Megatron 对应文件，只记接口边界，不陷入全部实现。
2. 写 45-60 分钟自学框架代码。
3. 跑 15 分钟最小测试或 smoke run。
4. 写 10 分钟学习日志：今天复刻了什么、Megatron 为什么这么设计、还没懂什么。

## 3. 第一阶段代码骨架

建议不要直接在 Megatron 源码目录里混写训练框架。新建独立目录，例如：

```text
mini-megatron/
├── configs/
│   └── gpt_tiny.yaml
├── minim/
│   ├── __init__.py
│   ├── config.py
│   ├── train.py
│   ├── state.py
│   ├── schedules.py
│   ├── distributed.py
│   ├── parallel_state.py
│   ├── core/
│   │   ├── model.py
│   │   └── transformer.py
│   ├── data/
│   │   └── text_dataset.py
│   └── checkpointing.py
└── tests/
    ├── test_model_shapes.py
    ├── test_checkpointing.py
    └── test_gradient_accumulation.py
```

## 4. 第一周任务清单

- [ ] 建立独立 repo 或独立目录。
- [ ] 写 `GPTConfig` dataclass。
- [ ] 写 `CausalSelfAttention`，确保 causal mask 正确。
- [ ] 写 `TransformerBlock` 和 `GPTModel`。
- [ ] 写 tiny text dataset。
- [ ] 写单卡训练 loop。
- [ ] 保存和恢复 checkpoint。
- [ ] 添加 shape/backward/checkpoint 三个测试。

## 5. 验收命令建议

单卡 smoke run：

```bash
python -m minim.train --config configs/gpt_tiny.yaml
```

DDP smoke run：

```bash
torchrun --nproc-per-node 2 -m minim.train --config configs/gpt_tiny.yaml
```

测试：

```bash
pytest -q tests
```

## 6. 学习时优先读的 Megatron 文件

第一轮只读这些，够用：
- `examples/run_simple_mcore_train_loop.py`
- `megatron/core/QuickStart.md`
- `megatron/training/training.py`
- `megatron/core/parallel_state.py`
- `megatron/core/pipeline_parallel/schedules.py`
- `megatron/core/tensor_parallel/layers.py`
- `megatron/core/datasets/gpt_dataset.py`
- `megatron/training/checkpointing.py`

读法：
- 先读函数签名和调用关系。
- 再看张量 shape 和通信位置。
- 最后才看性能优化、特殊 flags、兼容分支。

## 7. 暂不实现的内容

这些很重要，但放到第二阶段：
- MoE / expert parallel。
- FP8 / TransformerEngine。
- distributed optimizer / ZeRO。
- activation recomputation。
- sequence/context parallel。
- 异步 checkpoint。
- 通信计算重叠。

原因：这些是 Megatron 的规模化核心，但会遮住训练框架最基本的骨架。先把框架闭环跑顺，再逐个引入。

