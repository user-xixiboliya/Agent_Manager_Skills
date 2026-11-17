# 调整 batch size 和相关参数

> train-preflight 的参考文件。阶段 D 读。

## 基本原则

主要依据 Google《Deep Learning Tuning Playbook》中“Choosing the batch size”一节：

- batch size 决定训练速度，不应该拿来直接调验证集指标。通常最合适的 batch size 就是硬件能支持的最大值。
- 能放进显存的 batch size 很难事先算准。最简单的办法是按 2 的幂逐步增大，每个值跑少量步数，直到超出显存。
- 加大 batch 的好处都建立在吞吐增长的前提上。没有饱和时，batch 翻倍，吞吐也应该接近翻倍，每步耗时基本不变；做不到，说明有 I/O 或多卡同步之类的瓶颈，应该先查清。吞吐只增长到某个 batch 为止时，只考虑不超过这个值的 batch，即使显存还放得下更大的。
- 改了 batch size，学习率、动量等优化器参数和正则化参数通常都要重新调。
- 梯度累积只是模拟更大的 batch，不会提高吞吐。

落到本规范中：

- 每卡 batch 往大调，是为了吞吐；目标是在显存目标以内、吞吐还在增长的范围里取最大值。
- 等效 batch 是否跟着变，属于会改变训练结果的决定，必须问用户。用户要求等效 batch 不变时，用梯度累积补足，这时梯度累积是为了保持等效 batch，而不是为了提速。

## 调整顺序

### 第一步：确定哪些不能动

在确认点 1 中已经问过用户哪些参数允许调整。没有说明的，默认等效 batch、学习率及其调度、精度、序列长度或分辨率、优化器都保持不变。

### 第二步：用 2 的幂找上限

1. 从当前的每卡 batch 开始（太小时从 1 开始），按 2 的幂逐步加倍。
2. 每个值用最坏情况的 batch 跑训练、评估、保存三个阶段，按 smoke-test.md 的“第二级”记录峰值；再用随机 batch 按“第三级”测吞吐。
3. 每个值单独起一个进程测试。一次 OOM 之后，同一个进程里的显存分配状态已经被打乱，接着测的结果不可靠。
4. 每个值的结果（每卡 batch、三个阶段的峰值、nvidia-smi 进程峰值、吞吐、是否 OOM）写进结果目录的 `sweep.json`。
5. 出现以下任一情况就停止加倍：OOM；峰值超过显存目标；batch 翻倍但吞吐增长很少。

PyTorch Lightning 的 batch size finder 也是这个思路：`power` 模式不断加倍直到 OOM，`binsearch` 模式在加倍遇到 OOM 后再做二分查找；每次尝试默认跑 3 步（`steps_per_trial`）。项目用 Lightning 时可以直接用它找上限，但要注意：它目前不支持 DDP；它以训练步是否 OOM 为界，不留显存余量，找到的值通常要再往下调；评估和保存阶段的峰值仍要按 smoke-test.md 另外测。

### 第三步：在上限附近细找

在最后一个满足要求的值和第一个不满足的值之间二分，找出峰值不超过显存目标的最大每卡 batch。

- 吞吐已经在更小的 batch 上停止增长时，不需要细找，取吞吐停止增长处的值。
- 细找出来的值不一定是 2 的幂。和吞吐更好的 2 的幂相差不大时，可以取 2 的幂，在报告中写明取舍。

### 第四步：保持等效 batch

用户要求等效 batch 不变时：

```text
梯度累积步数 = 等效 batch ÷ (每卡 batch × 卡数)
```

- 除不尽时，不自行取整，把最接近的几种组合和对应的等效 batch 列给用户选。
- 检查训练代码在梯度累积时是否正确：loss 要除以累积步数（或者框架已经处理）；学习率调度和日志的步数按优化器更新次数算，而不是按前向次数算；多卡时，累积中间的几步可以用 `no_sync()` 跳过梯度同步。
- 用了 BatchNorm 时，BatchNorm 的统计量只看每卡 batch，梯度累积不会让它变大，见下文“注意事项”。

### 第五步：显存仍然不够时

按下表从上往下考虑，注意每项属于哪一类：

| 手段 | 作用 | 类别 |
|---|---|---|
| 单独设置评估 batch | 评估不算梯度，可以和训练用不同的 batch | 只影响速度和显存 |
| 梯度检查点 | 只保存部分激活值，反向时重新计算，用计算时间换显存 | 只影响速度和显存（`torch.utils.checkpoint` 默认保存并恢复随机数状态，dropout 不受影响） |
| 显存分配器设置 | 减少碎片，见下文 | 只影响速度和显存 |
| 混合精度（bf16、fp16） | 激活值和部分权重副本变小 | 会改变训练结果 |
| 更省显存的优化器（如 8-bit Adam） | 优化器状态变小 | 会改变训练结果 |
| 缩短序列、降低分辨率 | 激活值变小 | 会改变训练结果 |
| ZeRO、FSDP 等分片方式 | 优化器状态等分到多张卡 | 一般不改变结果，但改动大，必须先问用户 |

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
- PyTorch，CUDA semantics 中的“Memory management”“Optimizing memory usage with PYTORCH_ALLOC_CONF”两节：https://docs.pytorch.org/docs/stable/notes/cuda.html
- Goyal et al., 2017，《Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour》：https://arxiv.org/abs/1706.02677
- Malladi et al., 2022，《On the SDEs and Scaling Rules for Adaptive Gradient Algorithms》：https://arxiv.org/abs/2205.10287
