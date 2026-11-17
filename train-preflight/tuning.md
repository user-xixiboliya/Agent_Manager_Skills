# 在显存目标以内把吞吐调到最高

> train-preflight 的参考文件。阶段 D 读。

## 目标

阶段 D 要做的，是在显存目标以内让吞吐（每秒处理的样本数）尽量高。能调的东西有好几类，每卡 batch 只是其中一类：

| 调什么 | 主要影响 | 在哪一步 |
|---|---|---|
| 数值精度（混合精度） | 激活值显存、计算速度 | 第二步 |
| 每卡 batch | 激活值显存、吞吐 | 第三步 |
| 激活值怎么保存（梯度检查点） | 激活值显存、计算量 | 第四步 |
| 模型状态放在哪（更省显存的优化器、ZeRO/FSDP 分片、offload） | 权重、梯度、优化器状态这部分固定占用；通信 | 第四步 |
| 数据加载 | GPU 是否在等数据 | 第六步 |
| 显存分配器 | 碎片 | 第七步 |

这些设置互相影响：精度和梯度检查点决定了每卡 batch 能开多大，每卡 batch 又影响吞吐。所以顺序是先定精度这类基础设置，再找每卡 batch；显存限制了吞吐时，再回头加省显存的手段。每改一处都重新实测峰值和吞吐，最后比较的是显存目标以内的吞吐，选吞吐最高的组合，而不是显存最省的组合。

反过来也不要过度优化。HuggingFace 的单卡训练指南提醒：想要的配置本来就放得下时，不需要再加省显存的手段，梯度检查点、offload 这类手段会拖慢训练。

## 调整顺序

### 第一步：确定哪些不能动

在确认点 1 中已经问过用户哪些参数允许调整。没有说明的，默认等效 batch、学习率及其调度、精度、序列长度或分辨率、优化器都保持不变。

### 第二步：先定精度和几乎无代价的设置

这些设置会改变显存占用和速度，要在找每卡 batch 之前定下来，否则找到的 batch 上限换了设置后就不准了。

- **混合精度**：项目还没用时，在支持 bf16 的卡上通常是最先推荐的一步：多数情况下既提速，又省激活值显存。但它改变数值精度，属于会改变训练结果的修改，要问用户。
  - 按 HuggingFace 文档，节省主要来自激活值以半精度保存，这部分大致减半；梯度会转回全精度做更新，这部分不省；模型大、batch 小时，显存里同时有半精度和全精度两份权重，占用反而可能更多。PyTorch 的 `torch.autocast` 不另存半精度权重（见 estimation.md）。
  - bf16 的数值范围和 fp32 相同，需要 Ampere 及以上的卡，用 `torch.cuda.is_bf16_supported()` 确认；fp16 范围小，要配合 `GradScaler` 防止梯度下溢。
  - 冒烟测试和正式训练的精度设置必须一致。
- **评估 batch**：评估不算梯度，可以和训练用不同的 batch。评估阶段峰值最高时，单独调小评估 batch，不影响训练结果。
- **显存分配器**：碎片明显时（reserved 比 allocated 高出很多）按第七步设置。

### 第三步：找每卡 batch 的上限

每卡 batch 的作用是提高吞吐。按 Google《Deep Learning Tuning Playbook》中“Choosing the batch size”一节：

- 通常最合适的是硬件能支持、吞吐还在增长的最大值。没有饱和时，batch 翻倍，吞吐也应该接近翻倍；做不到，说明有 I/O 或多卡同步之类的瓶颈，先按第六步查清。吞吐不再增长时，即使显存还放得下，也不再加大。
- batch size 不用来调验证集指标。改了等效 batch，学习率、动量和正则化参数通常都要重新调，见“学习率和 batch 的关系”。
- 能放下多大很难事先算准，要实测。

做法：

1. 从当前的每卡 batch 开始（太小时从 1 开始），按 2 的幂逐步加倍。
2. 每个值用最坏情况的 batch 跑训练、评估、保存三个阶段，按 smoke-test.md 的“第二级”记录峰值；再用随机 batch 按“第三级”测吞吐。
3. 每个值单独起一个进程测试。一次 OOM 之后，同一个进程里的显存分配状态已经被打乱，接着测的结果不可靠。
4. 出现以下任一情况就停止加倍：OOM；峰值超过显存目标；batch 翻倍但吞吐增长很少。
5. 在最后一个满足要求的值和第一个不满足的值之间二分细找。吞吐已经停止增长时不用细找。细找出来的值不一定是 2 的幂，和吞吐更好的 2 的幂相差不大时可以取 2 的幂，在报告中写明取舍。
6. 每个值的结果（每卡 batch、三个阶段的峰值、nvidia-smi 进程峰值、吞吐、是否 OOM）写进结果目录的 `sweep.json`。

停下来时吞吐还在增长，说明显存限制了吞吐；每卡 batch 为 1 也放不下时更是如此。这时进入第四步。

项目用 PyTorch Lightning 时，可以用它的 batch size finder 找上限：`power` 模式不断加倍直到 OOM，`binsearch` 模式在加倍遇到 OOM 后再二分。注意它目前不支持 DDP；它以训练步是否 OOM 为界，不留显存余量；评估和保存阶段的峰值仍要按 smoke-test.md 另外测。

### 第四步：显存限制了吞吐时，从轻到重加手段

先看显存主要花在哪里：权重、梯度、优化器状态这部分固定占用按 estimation.md 算出来，实测峰值减去它，大致就是激活值和临时张量。激活值占大头时，梯度检查点、缩短序列有效；固定占用占大头时，要靠更省显存的优化器、分片或 offload。

下表按代价从轻到重排列，大多可以和第二步的混合精度叠加，也可以互相叠加：

| 顺序 | 手段 | 省的是哪部分 | 代价 | 类别 |
|---|---|---|---|---|
| 1 | 梯度检查点 | 激活值：只保存部分中间结果，反向时重新计算 | 训练变慢，HuggingFace 文档给的经验值约 20% | 只影响速度和显存 |
| 2 | 较小的每卡 batch 加梯度累积 | 本身不省显存，峰值由每卡 batch 决定；作用是在每卡 batch 变小后保持等效 batch | 每卡 batch 越小、累积步数越多，吞吐降得越多 | 保持等效 batch 时只影响速度和显存 |
| 3 | 更省显存的优化器（如 8-bit Adam） | 优化器状态 | 依实现而定 | 会改变训练结果 |
| 4 | ZeRO、FSDP 分片（仅多卡） | 优化器状态、梯度、参数，分到各张卡上 | 通信增加；改动大 | 一般不改变结果，但必须先问用户 |
| 5 | Offload：把优化器状态、参数换到 CPU 内存或 NVMe | 优化器状态、参数 | CPU 和 GPU 之间的传输成为瓶颈，明显变慢；还占内存或硬盘 | 必须先问用户，作为最后手段 |
| 6 | 缩短序列、降低分辨率 | 激活值 | 训练的内容变了 | 会改变训练结果 |

各项的要点：

- **梯度检查点**：Chen 等人 2016 年的论文《Training Deep Nets with Sublinear Memory Cost》提出，n 层的网络每隔约 √n 层存一次，激活值显存可以降到 √n 量级，代价是每个 batch 大约多一次前向计算；论文中 1000 层的 ResNet 从 48G 降到 7G，运行时间多约 30%。它只省激活值，固定占用不变。是否值得用，看实测：省出的显存换成更大的每卡 batch 后，吞吐的提升能不能抵过重算的损失；或者不开就放不下。`torch.utils.checkpoint` 默认保存并恢复随机数状态，dropout 不受影响；调用时按所用 torch 版本的文档明确传 `use_reentrant`。
- **梯度累积**：它不降低峰值，也不提高吞吐，作用是在每卡 batch 受显存限制时保持等效 batch，做法见第五步。每卡 batch 仍要取放得下的最大值。HuggingFace 文档的例子：每卡 batch 4 已经到显存上限、想要等效 batch 64 时，用 4 × 16，而不是 1 × 64。
- **ZeRO、FSDP**：普通的多卡数据并行（DDP）中，每张卡都存一份完整的参数、梯度和优化器状态。ZeRO 论文给出：只分片优化器状态（Stage 1），这部分模型状态的显存约降到 1/4；再分片梯度（Stage 2）约降到 1/8，这两级通信量和普通数据并行相同；再分片参数（Stage 3），显存随卡数成比例下降，通信量约为 1.5 倍。这些比例是论文按混合精度 Adam 和较多卡数算的，只针对模型状态，激活值不受影响，激活值大时还要配合梯度检查点。FSDP 是 PyTorch 自带的实现，计算前把分片的参数收集齐，算完再分开。单卡上分片没有作用。
- **Offload**：DeepSpeed 的 ZeRO-Infinity 可以换到 CPU 内存和 NVMe，FSDP 也支持换到 CPU。ZeRO 论文提到，此前把模型状态放到 CPU 的方法，最多有 50% 的训练时间花在 GPU 和 CPU 之间的传输上。用之前按 env-first 查清内存和硬盘是否够用。

每加一项手段，都按 smoke-test.md 的“第二级”“第三级”重新测峰值和吞吐，记进 `sweep.json`，再回到第三步，看省出来的显存能不能换成更大的每卡 batch。考虑过但没用的手段，也在报告中写明原因。

### 第五步：保持等效 batch

用户要求等效 batch 不变时，用梯度累积补足：

```text
梯度累积步数 = 等效 batch ÷ (每卡 batch × 卡数)
```

- 除不尽时，不自行取整，把最接近的几种组合和对应的等效 batch 列给用户选。
- 检查训练代码在梯度累积时是否正确：loss 要除以累积步数（或者框架已经处理）；学习率调度和日志的步数按优化器更新次数算，而不是按前向次数算；多卡时，累积中间的几步可以用 `no_sync()` 跳过梯度同步。
- 用了 BatchNorm 时，BatchNorm 的统计量只看每卡 batch，梯度累积不会让它变大，见下文“注意事项”。

### 第六步：数据加载

GPU 利用率偏低、等数据的时间占比高时，调整 DataLoader：

| 参数 | 作用 | 注意 |
|---|---|---|
| `num_workers` | 用多个子进程准备数据 | 不超过可用的 CPU 核数；共用机器上不要占满 CPU；每个 worker 都占内存 |
| `pin_memory=True` | 锁页内存，加快拷贝到 GPU | 占用更多内存 |
| `prefetch_factor` | 每个 worker 提前准备的 batch 数 | 只在 `num_workers > 0` 时有效 |
| `persistent_workers=True` | 每个 epoch 之间不重建 worker | 只在 `num_workers > 0` 时有效 |

- 在 docker 中运行时，worker 通过共享内存传数据，`/dev/shm` 太小会报错，用 `df -h /dev/shm` 查看。
- 改 worker 数会改变数据增强的随机序列，如果项目要求严格复现，问用户。

### 第七步：显存分配器设置

按 PyTorch CUDA 文档。较新版本的文档把环境变量的名字写作 `PYTORCH_ALLOC_CONF`，并说明 `PYTORCH_CUDA_ALLOC_CONF` 是为了向后兼容保留的别名；较旧的版本只认 `PYTORCH_CUDA_ALLOC_CONF`。所以统一用 `PYTORCH_CUDA_ALLOC_CONF`，并以项目所用 torch 版本的文档为准。

- `expandable_segments:True`：让分配器先建一段显存，需要时再往后扩展，而不是每次分配都新建一段，可以减少碎片。reserved 比 allocated 高出很多、或者 batch 中张量大小变化频繁时可以试。
- `max_split_size_mb`：禁止拆分大于这个大小的块，可以减少碎片，但性能代价可能从零到很大。文档说明这只应作为最后手段，用于出现大量未用的拆分块而 OOM 的情况。
- 设置方式：只写在启动命令或项目的启动脚本里，例如 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py`，不写进 `~/.bashrc` 等全局配置。
- 改了设置后重新测峰值，冒烟测试和正式训练用同样的设置。

## 学习率和 batch 的关系

只有用户同意改变等效 batch 时，才需要考虑这一节。以下规则只是重新调参的起点，是否采用由用户决定：

- **SGD**：Goyal 等人 2017 年的论文《Accurate, Large Minibatch SGD》提出线性缩放规则：等效 batch 扩大 k 倍，学习率也扩大 k 倍；同时在训练初期用逐步升高学习率的 warmup。
- **Adam、RMSprop**：Malladi 等人 2022 年的论文《On the SDEs and Scaling Rules for Adaptive Gradient Algorithms》推导出平方根缩放规则：等效 batch 扩大 k 倍，学习率扩大 √k 倍。论文中的完整规则还涉及 Adam 的其他超参数，采用前读原文中的定义。
- 按 Tuning Playbook 的说法，换 batch size 后最稳妥的做法是重新调优化器参数。上面的规则不能代替重新调参，在报告中写明采用了哪条规则。

## 注意事项

- **BatchNorm**：BatchNorm 用当前 batch 的均值和方差做归一化，多卡时每张卡的统计量各不相同，除非显式同步。Tuning Playbook 提到，有经验报告称只用约 64 个样本计算统计量在实践中效果更好（Ghost BatchNorm）。每卡 batch 很小时统计量不稳；可以考虑 `SyncBatchNorm`，但这会改变训练结果，要问用户。另外，有的实现只保存第一张卡上的滑动平均统计量，保存 checkpoint 前要确认是否已同步。
- **变长输入**：按长度分桶，可以减少补齐带来的浪费；测峰值时必须包括最长的那个桶。分桶改变了 batch 的组成方式，属于会改变训练结果的修改，要问用户。
- **多卡**：只由一个进程写日志和保存 checkpoint；模型初始化的随机种子各卡相同，数据打乱和增强的随机种子各卡不同。
- **改了模型或优化器**：Tuning Playbook 提醒，模型结构或优化器变了以后，能放下的 batch size 也会变，要重新走一遍本文件的流程。

## 出处

- Google Research，《Deep Learning Tuning Playbook》，“Choosing the batch size”“Batch normalization implementation details”“Considerations for multi-host pipelines”各节：https://github.com/google-research/tuning_playbook
- PyTorch Lightning，“Batch Size Finder”：https://lightning.ai/docs/pytorch/stable/advanced/training_tricks.html
- HuggingFace Transformers，《Methods and tools for efficient training on a single GPU》中“Gradient Accumulation”“Gradient Checkpointing”“Mixed precision training”各节（v4.46 版本，新版文档已改版）：https://github.com/huggingface/transformers/blob/v4.46.0/docs/source/en/perf_train_gpu_one.md
- Chen et al., 2016，《Training Deep Nets with Sublinear Memory Cost》：https://arxiv.org/abs/1604.06174
- Rajbhandari et al., 2020，《ZeRO: Memory Optimizations Toward Training Trillion Parameter Models》：https://arxiv.org/abs/1910.02054
- DeepSpeed，“Zero Redundancy Optimizer”教程：https://www.deepspeed.ai/tutorials/zero/
- PyTorch，“Getting Started with Fully Sharded Data Parallel (FSDP2)”：https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
- PyTorch，CUDA semantics 中的“Memory management”“Optimizing memory usage with PYTORCH_ALLOC_CONF”两节：https://docs.pytorch.org/docs/stable/notes/cuda.html
- Goyal et al., 2017，《Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour》：https://arxiv.org/abs/1706.02677
- Malladi et al., 2022，《On the SDEs and Scaling Rules for Adaptive Gradient Algorithms》：https://arxiv.org/abs/2205.10287
