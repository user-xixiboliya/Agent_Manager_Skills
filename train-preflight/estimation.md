# 显存和硬盘的估算

> train-preflight 的参考文件。阶段 B 读。

估算的作用是在上卡之前排除明显不可行的配置，并给硬盘留出足够空间。最终采用的数字以冒烟测试实测为准。

## 显存由哪几部分组成

按 HuggingFace《Model memory anatomy》，训练时显存里主要有这几类东西：

| 组成 | 大小 | 说明 |
|---|---|---|
| 模型权重 | 混合精度下约 6 字节/参数 | 一份 fp16 副本用于前向和反向（2 字节），一份 fp32 主副本用于稳定地更新权重（4 字节） |
| 优化器状态 | Adam 约 8 字节/参数 | 动量和方差各一份，都是 fp32；bitsandbytes 的量化 Adam 可压到约 2 字节/参数 |
| 梯度 | fp32 约 4 字节/参数 | 反向时为每个参数算出 |
| 激活值 | 随 batch、序列长度、层数、隐藏维度增长 | 前向时缓存下来给反向用。模型本身放得下时，batch 或序列长度也可能把显存撑满 |
| 临时张量 | 单个算子的瞬时占用 | softmax、矩阵乘法等算子产生，用完释放；单个算子峰值很高时会造成瞬时 OOM |
| 其他开销 | 视功能而定 | 例如大词表的 embedding 表、beam search 的缓存 |

注意力分数矩阵随序列长度的平方增长，这是长序列特别占显存的原因。

此外还有不属于 PyTorch 张量的部分：CUDA 运行时自身的固定开销（CUDA context）、cuDNN 和 cuBLAS 的工作空间。这部分不在 PyTorch 的统计里，用 `nvidia-smi` 中本进程的占用减去 `torch.cuda.max_memory_reserved()` 得到实测值，不凭经验填数字。

### 每个参数的固定占用

权重、梯度、优化器状态三项只和参数量有关，可以在上卡前算出来。下表按上面的单项大小相加，用之前先看项目实际用的精度和优化器：

| 训练方式 | 权重 | 梯度 | 优化器状态 | 合计（字节/参数） |
|---|---|---|---|---|
| fp32，Adam / AdamW | 4 | 4 | 8 | 16 |
| HuggingFace 描述的混合精度，Adam / AdamW | 6 | 4 | 8 | 18 |
| fp32，带动量的 SGD | 4 | 4 | 4 | 12 |
| fp32，不带动量的 SGD | 4 | 4 | 0 | 8 |

- PyTorch 的 `torch.autocast` 默认不另存一份 fp16 权重，权重仍是 fp32；用它时按 fp32 那一行估算，再以实测为准。
- 冻结的参数（`requires_grad=False`）没有梯度和优化器状态，只算权重。
- 参数量用 `sum(p.numel() for p in model.parameters())` 实际统计；需要区分可训练参数时，加上 `if p.requires_grad`。

例子：1 亿个参数，fp32 加 AdamW，固定占用约 1e8 × 16 字节 ≈ 1.6 GB。剩下的显存才能给激活值、临时张量和 CUDA 开销。

### 什么时候可以判定“放不下”

- 固定占用已经超过目标卡的可用显存：不用上卡试，直接告诉用户，并给出能减小固定占用的选项：更省显存的优化器、多卡分片（ZeRO、FSDP）、offload、换卡，做法和代价见 tuning.md 第四步，其中会改变训练结果的要说明。梯度检查点只省激活值，对固定占用没有帮助。
- 固定占用没超过：激活值要靠冒烟测试实测，不用公式硬算。激活值的估算公式和模型结构、实现方式关系很大，误差常常比要留的显存余量还大。

## 三种显存读数的区别

按 PyTorch CUDA 文档的“Memory management”一节：

| 读数 | 含义 |
|---|---|
| `torch.cuda.memory_allocated()`、`max_memory_allocated()` | 张量实际占用的显存（当前值、峰值） |
| `torch.cuda.memory_reserved()`、`max_memory_reserved()` | PyTorch 缓存分配器管理的显存总量，包括暂时没用、留着复用的部分 |
| `nvidia-smi` | 进程占用的全部显存。PyTorch 缓存着但暂时没用的显存也显示为已占用，另外还包括 CUDA context |

- `torch.cuda.empty_cache()` 只释放缓存里没用的部分，给别的程序用；不会让 PyTorch 自己能用的显存变多，不能用来解决 OOM。
- reserved 比 allocated 高出很多，说明有显存碎片，处理方法见 tuning.md。
- 判断离 OOM 还有多远，看 `nvidia-smi` 的进程峰值占卡的可用显存的比例。

## 多卡

- 等效 batch = 每卡 batch × 梯度累积步数 × 卡数。改卡数时，等效 batch 会跟着变。
- DDP：每张卡上都有完整的权重、梯度和优化器状态，另外还要给梯度通信留缓冲区。按单卡估算后，以实测为准。
- ZeRO、FSDP：会把优化器状态（以及按配置把梯度、权重）分到各张卡上。每张卡按分片后的大小估算，具体分哪几项以项目配置为准。
- 卡的型号或已被占用的显存不一样时，按最紧张的那张卡定每卡 batch。

## 硬盘

### 单个 checkpoint 多大

- PyTorch 保存与加载教程中的说法：用于续训的 checkpoint 除了模型，还要存优化器状态等内容，通常是只存模型时的 2–3 倍。
- 按参数量估算：权重按保存时的精度算（fp32 是 4 字节/参数），Adam 状态再加约 8 字节/参数。例如 1 亿个参数、fp32、AdamW，单个续训 checkpoint 约 1e8 × 12 字节 ≈ 1.2 GB。
- 只存权重的文件（用于评估或发布）单独算。
- 分片保存（DeepSpeed、FSDP）时，所有分片加起来算一个 checkpoint。
- 冒烟测试保存过 checkpoint 后，用 `du -sh <checkpoint 路径>` 拿到实测大小，替换估算值。

### 整个训练要多少

```text
总量 = 单个续训 checkpoint × 保留个数
     + 单个续训 checkpoint × 1        # 先写临时文件再改名，写入时会多出一份
     + 只存权重的文件 × 个数           # 例如 best、最终权重
     + 日志和 TensorBoard / wandb 本地文件
     + 评估输出（预测结果、可视化）
     + 数据预处理缓存
     + 编译缓存（torch.compile、triton、CUDA JIT）
```

- 日志和评估输出按冒烟测试中实际产生的大小，乘以正式训练的步数比例推算。
- 数据预处理缓存和编译缓存放在哪个盘，按“运行环境”中记录的缓存位置算；它们不能放进产物目录。
- 用 `df -h <产物目录所在的盘>` 看剩余空间，用 `du -sh` 看已有占用。别人也在用这块盘时，剩余空间会变化，在报告中注明。
- 总量超过剩余空间，或者超过剩余空间的一半时，在确认点 2 问用户。选项可以是：减少保留个数、优化器状态只保留最新一份、换到更大的盘。

## 出处

- HuggingFace Transformers，《Model memory anatomy》（GPU memory usage）：https://huggingface.co/docs/transformers/main/en/model_memory_anatomy
- PyTorch，CUDA semantics 中的“Memory management”一节：https://docs.pytorch.org/docs/stable/notes/cuda.html
- PyTorch，《Saving and Loading Models》中的“Saving & Loading a General Checkpoint”一节：https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
