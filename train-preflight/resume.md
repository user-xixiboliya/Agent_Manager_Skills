# 断点续训

> train-preflight 的参考文件。阶段 C 的续训测试和阶段 E 读。

断点续训的目标是：训练在任何时刻中断（机器重启、进程被杀、集群抢占），都能从最近一次保存的位置接着训练，效果和没有中断过一样。只存模型权重做不到这一点，PyTorch 保存与加载教程也指出，用于续训的 checkpoint 必须同时保存优化器状态等内容。

项目已经用了训练框架时，优先用框架自带的续训方式，见下文“常用框架”；下面的清单用来检查框架有没有漏掉什么。

## 要保存的内容

| 内容 | 为什么要存 | 常见写法 |
|---|---|---|
| 模型权重 | — | `model.state_dict()`；DDP 包装后用 `model.module.state_dict()` |
| 优化器状态 | Adam 的动量和方差等，不存会让恢复后的更新方向突变 | `optimizer.state_dict()` |
| 学习率调度器 | 不存会让学习率从头开始 | `scheduler.state_dict()` |
| AMP 的 GradScaler | 用 fp16 时保存当前的 loss 缩放系数 | `scaler.state_dict()` |
| EMA 权重 | 有 EMA 时，评估常用它 | EMA 对象的 `state_dict()` |
| 进度 | 知道从哪里接着训练 | epoch、全局步数（按优化器更新次数算）、本 epoch 已消耗的 batch 数 |
| 采样器和数据顺序 | 恢复后不重复、不跳过数据 | 采样器的随机种子和当前 epoch；`DistributedSampler` 每个 epoch 要调用 `set_epoch(epoch)` |
| 随机数状态 | 数据增强、dropout 的随机序列能接上 | `random.getstate()`、`np.random.get_state()`、`torch.get_rng_state()`、`torch.cuda.get_rng_state_all()`；DataLoader 用了 `generator` 时也要存 |
| 最佳指标 | 恢复后能继续判断“最好的 checkpoint” | 当前最佳指标值和对应步数 |
| 配置快照 | 恢复时检查配置是否一致 | 完整配置，或至少关键字段 |
| 代码版本 | 追溯这个 checkpoint 是哪版代码产生的 | `git rev-parse HEAD`，有未提交修改时注明 |
| 实验跟踪的 run id | 恢复后日志接在同一条曲线上 | 例如 wandb 的 run id |

多卡时每张卡的随机数状态不同。要求严格接上时，每个进程分别保存自己的随机数状态，或者按 rank 从同一个种子推出。

### 从 epoch 中间恢复

按时间保存时，保存点多半落在 epoch 中间，有两种做法：

- **记录已消耗的 batch 数，恢复后跳过**：实现简单。但跳过时如果要把前面的数据真的读一遍，数据量大时会很慢；采样器能直接从指定位置开始时更好。
- **只在 epoch 结束时保存**：最简单，但一个 epoch 很长时，中断会丢掉很多进度。

用哪种做法，结合 epoch 的时长在确认点 3 问用户。

## 怎么保存

- **先写临时文件，再改名**：先 `torch.save` 到同目录下的临时文件，写完后用 `os.replace` 改成正式文件名。中途中断只会留下临时文件，正式文件不会是半截的。
- **记录最新的 checkpoint**：保存成功后，用同样的“先写临时文件再改名”方式更新一个记录最新 checkpoint 路径的文件（例如 `latest`），自动续训读它。
- **多卡**：普通 DDP 只由 rank 0 写文件，写之前用 `barrier()` 让各进程对齐。DeepSpeed、FSDP 的分片保存要所有进程一起调用框架的保存接口，按框架文档来，不能只在 rank 0 调用。
- **保留规则**：Tuning Playbook 建议在训练中保留目前为止最好的 N 个 checkpoint，训练结束后从中挑选，不要只依赖最后一个。默认保留最近几个用于续训，再加上最好的几个用于挑选。训练程序按保留规则自动删除旧 checkpoint，属于删除实验产物，在确认点 4 由用户确认保留个数；agent 自己不手动删除任何 checkpoint。
- **保存间隔按时间定**：先和用户商定中断时最多能接受丢掉多长时间的进度（例如 30–60 分钟），再用冒烟测试测出的每步耗时换算成步数：

  ```text
  保存间隔（步） = 可接受丢失的时间（秒） ÷ 每步耗时（秒）
  ```

  保存本身也要时间，在冒烟测试中实测一次保存的耗时；保存太频繁会明显拖慢训练时，在报告中写明取舍。
- **被抢占时先保存**：集群在结束任务前会先发信号（例如 slurm 的 SIGTERM，或用 `--signal` 指定的信号）。需要时可以捕获信号，保存后再退出；这会改动训练入口，先在确认点 3 问用户。

## 怎么恢复

### 两种方式

| 方式 | 找不到 checkpoint 时 |
|---|---|
| 指定路径（例如 `--resume <路径>`） | 直接报错退出，报错中写明找的路径。不能悄悄从头开始 |
| 自动找最新（例如 `--resume auto`） | 读 `latest` 记录的路径。输出目录里确实一个 checkpoint 都没有时，可以从头训练，但要打印明确的日志，写明找过哪个目录；`latest` 存在但指向的文件读不出来时，直接报错 |

不设置续训参数时是否默认自动续训，问用户。

### 恢复的顺序

1. 构建模型，移到对应的设备上，再包装 DDP 等。
2. 构建优化器、学习率调度器、GradScaler。
3. 加载 checkpoint，依次恢复模型、优化器、调度器、GradScaler、EMA、进度、随机数状态。
4. 按进度设置采样器（`set_epoch`）和要跳过的 batch 数。
5. 恢复后的第一步打印：从哪个文件恢复、恢复到第几步、当前学习率。

### 检查配置是否一致

恢复时，把 checkpoint 中的配置快照和当前配置比较。下面这些字段不一致时，直接报错，列出哪些字段不一致：

- 模型结构相关的配置
- 等效 batch 及其组成（每卡 batch、梯度累积、卡数）
- 优化器种类、学习率调度的总步数
- 数据集和数据预处理

用户明确要求改配置后接着训练时（例如换了卡数），由用户决定怎么处理，并在日志里写明改了什么。不自动改配置去迁就 checkpoint，也不忽略不一致继续训练。

### 加载时的注意事项

- torch 2.6 起，`torch.load` 默认 `weights_only=True`，只允许加载张量和基本类型。checkpoint 里存了 `np.random.get_state()` 这类对象时会加载失败。可以把随机数状态转成张量或基本类型再存；或者对自己训练产生、来源可信的文件显式传 `weights_only=False`。来源不明的文件不能用 `weights_only=False` 加载，它会执行文件中的任意代码。
- 用 `map_location` 指定加载到哪个设备，避免所有进程都把 checkpoint 加载到 0 号卡上，造成 0 号卡显存突增。

## 常用框架

| 框架 | 保存 | 恢复 | 需要额外确认的 |
|---|---|---|---|
| PyTorch Lightning | `ModelCheckpoint` 回调，`save_last=True` 保存最新一个 | `trainer.fit(model, ckpt_path="last")`，或传具体路径 | 自定义的状态（如 EMA）是否通过 `on_save_checkpoint` / `on_load_checkpoint` 存取 |
| HF Trainer | `save_strategy`、`save_steps`，`save_total_limit` 控制保留个数 | `trainer.train(resume_from_checkpoint=True)` 找输出目录中最新的，或传具体路径 | 输出目录里有没有旧运行留下的 checkpoint |
| DeepSpeed | `engine.save_checkpoint(save_dir, tag)` | `engine.load_checkpoint(load_dir, tag)` | 所有进程都要调用；自定义状态通过 `client_state` 存取 |

- 框架在找不到 checkpoint 时的行为可能随版本变化，按项目所用版本实测确认，必须符合上面“两种方式”中的规则；不符合时在框架外加一层检查。
- 框架自带的保存已经覆盖了优化器、调度器等内容，但项目自己加的状态（EMA、自定义采样器、额外的随机数生成器）往往需要手动接入，对照“要保存的内容”逐项检查。

## 怎么验证续训

在冒烟测试中做，结果写进最终报告：

1. **连续训练**：从同一个起点、同样的随机种子出发，连续训练 2N 步，记录每一步的 loss 和学习率，得到 A。
2. **中断后恢复**：同样的起点和种子，训练 N 步并保存；用 `kill -9 <PID>` 结束进程，模拟意外中断，而不是正常退出；再用续训参数启动，训练到 2N 步，得到 B。
3. **对比**：
   - 恢复后的步数、epoch、学习率必须和 A 在同一步上完全一致。
   - loss 曲线：完全确定性的设置下应该一致；开了 `cudnn.benchmark`、用了 FlashAttention 等反向不确定的算子时，会有小的数值差异。判断标准（例如每步 loss 的相对差异不超过多少）在报告中写明，并说明差异来自哪里。
   - 有 GradScaler 时，恢复后的缩放系数和 A 一致。
4. **检查找不到 checkpoint 的行为**：指定一个不存在的路径，确认直接报错；在空目录上用自动续训，确认打印了明确的日志后从头开始。
5. **检查半截文件**：确认保存中途被打断留下的临时文件不会被当作最新的 checkpoint。

PyTorch 的随机性说明文档中列出了影响确定性的设置（`torch.manual_seed`、`cudnn.benchmark`、`torch.use_deterministic_algorithms` 等）。为了验证续训而开启确定性设置时，只在验证用的运行中开，正式训练是否开由用户决定，因为可能降低速度。

## 出处

- PyTorch，《Saving and Loading Models》中的“Saving & Loading a General Checkpoint for Inference and/or Resuming Training”一节：https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html
- PyTorch，《Reproducibility》：https://docs.pytorch.org/docs/stable/notes/randomness.html
- Google Research，《Deep Learning Tuning Playbook》，“Saving checkpoints and retrospectively selecting the best checkpoint”一节：https://github.com/google-research/tuning_playbook
