# 冒烟测试

> train-preflight 的参考文件。阶段 C 读；阶段 D 每次调整参数后按“第二级”重新测量。

冒烟测试是用很少的步数把正式训练的整条路径走一遍，目的有三个：确认能跑通，实测显存峰值，实测吞吐。续训测试也在冒烟测试中完成，做法见 resume.md。

## 测试条件要和正式训练一致

冒烟测试的结果只有在条件一致时才能用来定正式训练的参数。以下各项必须和正式训练相同：

- 同一台机器、同样的卡号、同样的卡数和并行方式（单卡、DDP、FSDP、DeepSpeed）
- 同样的代码版本（记下 `git rev-parse HEAD`，有未提交的修改时注明）
- 同样的模型、精度、优化器、梯度检查点设置、`torch.compile` 设置
- 同样的环境变量，尤其是 `PYTORCH_CUDA_ALLOC_CONF`、`CUDA_VISIBLE_DEVICES`
- 同样的数据预处理和数据增强

只能改：总步数、评估和保存的频率（改小，以便在几步内触发）、输出目录。

开始前用 `nvidia-smi` 看目标卡上别的进程占用了多少，记进结果。不使用用户没允许的卡，不结束别人的进程。

## 第一级：跑通

用少量步数（例如 10–20 步）把以下环节都走一遍，任何一环报错都先修好再往下：

1. 数据加载：能取出 batch，形状和类型符合预期
2. 前向、反向、优化器更新：loss 是有限值，不是 NaN 或 Inf
3. 学习率调度：学习率按预期变化
4. 评估：能完整跑完一次评估
5. 保存：checkpoint 能写到磁盘，记下实测大小（`du -sh`）
6. 加载：能从刚保存的 checkpoint 恢复，并继续训练几步

报错时读完整报错，找第一个错误，针对原因处理，不随手换版本或换参数。

## 第二级：测显存峰值

### 用最坏情况的 batch

显存峰值由最占显存的那个 batch 决定，随机取几个 batch 测不出来。

- 变长输入（文本、语音、点云）：从真实数据中挑出最长的样本组成 batch。数据管道把所有样本都补齐到固定长度时，任意 batch 都是最坏情况。
- 图像：按数据增强后可能出现的最大分辨率；多尺度训练时取最大的尺度。
- 样本长度没有上限时，无法覆盖最坏情况，属于 SKILL.md 中“显存目标”一节说的需要多留余量的情况，和用户商量余量时说明这一点；同时告诉用户可以考虑加长度上限（这会改变训练结果，要用户决定）。

这些 batch 只用来测显存，不用来得出训练结论。

### 分阶段记录

训练、评估、保存三个阶段的峰值可能差别很大，要分别记录。每个阶段开始前清零峰值统计：

```python
import torch

def report_peak(tag, device=None):
    torch.cuda.synchronize(device)
    gib = 1024 ** 3
    print(
        f"[preflight] {tag}: "
        f"max_allocated={torch.cuda.max_memory_allocated(device) / gib:.2f} GiB, "
        f"max_reserved={torch.cuda.max_memory_reserved(device) / gib:.2f} GiB",
        flush=True,
    )

torch.cuda.reset_peak_memory_stats()
# ... 用最坏 batch 训练若干步 ...
report_peak("train")

torch.cuda.reset_peak_memory_stats()
# ... 跑一次完整评估 ...
report_peak("eval")

torch.cuda.reset_peak_memory_stats()
# ... 保存一次 checkpoint ...
report_peak("save")
```

把这段代码接进项目的训练入口前，先到达确认点 3。用了 Lightning、HF Trainer 等框架时，放进框架的回调里，不另写一套训练循环。

需要注意的几点：

- 训练阶段至少跑 3 步再读峰值。Adam 等优化器的状态在第一次 `optimizer.step()` 时才分配，`cudnn.benchmark`、`torch.compile` 也会在前几步额外占用显存。
- 用了梯度累积时，至少跑满一个完整的累积周期。
- 评估阶段的 batch size 可能和训练不同，按评估实际用的值测。
- 保存阶段：DeepSpeed、FSDP 把分片汇总成完整权重时，可能在某张卡上临时多占显存，要实测。
- 多卡时每张卡分别记录，按最高的那张卡判断。

### 同时记录 nvidia-smi

PyTorch 的统计不包括 CUDA context 等开销，所以另开一个终端在测试期间采样：

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv -lms 500 > <结果目录>/nvidia_smi_apps.csv
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -lms 500 > <结果目录>/nvidia_smi_gpu.csv
```

测试结束后停掉采样，从中找出本进程（按 PID）的最大 `used_memory`。这个值除以卡的可用显存，就是和显存目标比较的占比。

### 结果怎么看

| 现象 | 说明 | 怎么办 |
|---|---|---|
| 三个阶段中最高的峰值 ≤ 显存目标 | 当前配置可用 | 进入阶段 D，看吞吐还能不能提高 |
| 峰值 > 显存目标但没有 OOM | 余量不够，正式训练中遇到更长的样本或碎片时可能 OOM | 按 tuning.md 第二步、第四步从轻到重选手段（混合精度、梯度检查点等）；缩小每卡 batch 并加大梯度累积是其中一种，不是唯一的一种 |
| reserved 比 allocated 高出很多 | 显存碎片 | 见 tuning.md 中的分配器设置 |
| 评估阶段的峰值最高 | 评估的 batch 太大 | 单独减小评估 batch，不影响训练结果 |

## 第三级：测吞吐

- 先跑若干步热身（例如 10–20 步），不计入时间，避开 `cudnn.benchmark`、`torch.compile`、数据加载预热的影响。
- 计时前后调用 `torch.cuda.synchronize()`，否则测到的只是 CPU 提交任务的时间。
- 计时至少几十步，记录每秒样本数和每步耗时；用真实数据的随机 batch 测，不用最坏 batch。
- 同时看 `nvidia-smi` 中的 GPU 利用率。利用率明显偏低时，先查数据加载：分别记录每步“等数据”的时间和“计算”的时间，等数据的时间占比高，说明瓶颈在数据加载，调大 batch 也不会变快。
- 用吞吐推算正式训练时长：总步数 × 每步耗时，再加上评估和保存的时间。

## 产物放在哪里

- 先按项目已有的产物规范存放（AGENTS.md、CLAUDE.md、README、已有目录结构、训练框架的输出目录配置）。
- 项目没有规范时，放在产物根目录下单独的 `preflight/` 子目录中，和正式训练分开：

  ```text
  <产物根目录>/
    preflight/
      <实验名>/
        <YYYYMMDD-HHMMSS>/
          config.yaml            # 本次测试的完整配置快照
          train.log              # 训练日志，包括 [preflight] 开头的峰值记录
          nvidia_smi_apps.csv    # nvidia-smi 采样
          nvidia_smi_gpu.csv
          sweep.json             # 阶段 D 中每个 batch size 的峰值和吞吐
          checkpoints/           # 续训测试保存的 checkpoint
  ```

- 找不到产物根目录，或者有多个候选时，问用户放在哪里。
- 每次测试用新的时间戳目录，不覆盖已有的目录。
- 同一次测试的配置、日志、采样、checkpoint 放在同一个目录里，不拆散。
- 测试结束后不删除这些产物，在最终报告中列出路径和大小，交给用户处理。
- 临时写的测试脚本放在 `.agents/tmp/`，不放进源码目录。
