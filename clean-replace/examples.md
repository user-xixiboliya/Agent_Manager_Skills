# 完整示例：ResNet 编码器替换为 ViT

> clean-replace 的参考文件。需要参考完整流程时读。

## 场景

用户输入：

```text
/clean-replace ResNet 编码器实验效果不好，换成 ViT
```

## 错误做法

agent 执行了以下操作：

- 搜索 `ResNetEncoder`，只修改了命中的 3 个文件
- 新建 `models/vit_encoder.py`
- 把 `configs/default.yaml` 中的 `encoder: resnet` 改为 `encoder: vit`
- 为了防止 ViT 构建失败，加了 `try/except`，失败时退回 ResNet
- ViT 的 patch size、层数随手取了常见值
- 把旧结果目录改名为 `runs/resnet_old/`
- 报告完成

遗留问题：

- `models/encoder.py` 中的 `ResNetEncoder` 仍然存在
- `models/blocks.py` 中只服务于 ResNet 的 `ResNetBlock` 仍然存在
- `eval.py` 通过注册名 `"resnet"` 加载模型，搜索类名找不到它，没有被修改，评估时仍在用旧编码器
- `utils/head.py` 中写死了 ResNet 的输出维度 `2048`，没有出现旧名称，也被漏掉
- 新加的 fallback 让 ViT 出错时悄悄跑回 ResNet，实验结果无法区分
- `configs/resnet50.yaml` 仍然存在
- `tests/test_encoder.py` 仍在测试 ResNet
- README 和 `docs/method.md` 仍在描述 ResNet 和残差结构
- ViT 超参数没有来源，用户以为是官方设置
- 实验记录中没有记录这次替换
- agent 擅自重命名了用户的实验产物
- 整个过程没有向用户确认

结果：训练用的可能是 ViT，也可能悄悄退回了 ResNet；评估用的是 ResNet；文档描述的也是 ResNet；用户还找不到原来的结果目录。

## 正确做法

1. **阶段 A**
   - 读 AGENTS.md，得知产物目录是 `runs/`，项目没有指定修改索引的位置
   - 检查 `git status`
   - 找到 CHANGELOG、实验记录的格式
   - 写出替换关系，其中“新方法参数来源”标为“待确认”

2. **确认点 1**
   - 展示替换关系，询问 ViT 超参数的来源和预处理是否在范围内
   - 告诉用户：工作区有 2 个未提交文件，询问如何处理、是否先提交回退点
   - 告诉用户项目中还没有修改索引，将在 `docs/agent-index/` 创建

3. **阶段 B**
   - 没有修改索引，需要阅读 96 个文件，不超过 100 个，直接全部阅读
   - 逐个阅读后发现：
     - `eval.py` 通过字符串注册名间接引用旧编码器
     - `utils/head.py` 写死了 ResNet 的输出维度 `2048`
     - `registry.py` 查不到注册名时退回 `ResNetEncoder`
     - `docs/method.md` 用中文描述残差结构
     - `models/__init__.py` 公开导出了旧类，可能有外部调用方
     - AGENTS.md 写着“encoder 使用 ResNet”
   - 用补充后的旧标识交叉检查，命中项全部在已阅读文件中

4. **阶段 C 和确认点 2**
   - 先直白说明影响范围
   - 再给出扫描覆盖报告、受影响文件清单、已有 fallback 列表、CHANGELOG 草稿、实验记录草稿
   - 实验记录中“ResNet 效果较差”的具体数字，在结果文件中找不到依据，因此留空并询问用户
   - 输出后停下来等待确认

5. **阶段 D 到 F**
   - 按用户确认的范围原地替换，不写 fallback，构建失败时直接报错
   - 删除旧代码、fallback 和连带残留
   - 同步当前文档，用户确认后更新 AGENTS.md
   - 在 CHANGELOG 和实验记录中追加条目
   - 创建 `docs/agent-index/file-map.md`（96 行）和本次修改记录
   - 不触碰 `runs/resnet_encoder/`

6. **阶段 G**
   - 旧标识只剩 CHANGELOG、实验记录、修改记录和产物目录中的白名单命中
   - 没有新增备份文件，没有新增 fallback
   - 测试全部通过

7. **确认点 4**
   - 输出报告
   - 列出 `runs/resnet_encoder/`（3.2 GB）和根目录下散落的 `debug.log`，交给用户处理
   - 询问是否提交 commit

下次再替换编码器相关内容时，agent 可以根据 `file-map.md` 跳过内容没变、与编码器无关的文件，只阅读修改记录指向的文件、调用链上的文件和内容有变化的文件。
