# Agent_Manager_Skills

简体中文 | [English](README.en.md)

[![License](https://img.shields.io/badge/License-Apache_2.0-blue)](LICENSE)
![Domain](https://img.shields.io/badge/Domain-CS_Research-informational)
![Qoder](https://img.shields.io/badge/Qoder-supported-7B61FF)
![Claude Code](https://img.shields.io/badge/Claude_Code-supported-D97757)
![Codex](https://img.shields.io/badge/Codex-supported-10A37F)
![Docs](https://img.shields.io/badge/Docs-%E4%B8%AD%E6%96%87-red)

这里收集一些用来约束编程 agent 工作方式的 skill。每个 skill 放在单独的文件夹里，可以单独安装。

这些 skill 一般用于通用的计算机科研场景，例如在本机或 GPU 服务器上跑实验、训练模型、替换方法、整理实验记录和结果。

## Skill 列表

| Skill | 调用方式 | 用途 |
|---|---|---|
| [clean-replace](clean-replace/SKILL.md) | `/clean-replace` | 替换方法、模型、接口或实现路线时，彻底删除旧实现，同步更新所有引用、配置、测试和文档。禁止版本化副本、备份文件、兼容层和静默 fallback，也不写逐字段 isinstance 检查这类啰嗦的防御性代码。脚本参数保持精简，新增行为开关前先征得用户同意，默认行为由用户设定。修改前逐个文件扫描项目，并维护修改索引来缩小扫描范围。记录同一件事只写一处，状态文件只写当前值，文档不手抄哈希，临时文件放在 `.agents/tmp/`；也用于整理堆积了多个版本、重复记录和缓存的项目。 |
| [env-first](env-first/SKILL.md) | 安装、下载、GPU 相关任务时自动启用；也可以明确输入 env-first skill 时使用 | 动手前先查清机器环境（GPU、CUDA、Python 环境、镜像源和代理、磁盘、服务器的工作空间、外网访问和共用数据目录），写进项目的 AGENTS.md 或 CLAUDE.md。关于机器的结论必须有命令输出作为依据；下载失败先查目标是否存在、版本是否匹配；不新建多余的环境，不用 sed 改坏文件，不改全局配置，缓存不写进产物目录，镜像标签不复用。 |
| [train-preflight](train-preflight/SKILL.md) | `/train-preflight` | 正式训练前的准备：估算显存和硬盘占用；用冒烟测试实测训练、评估、保存三个阶段的显存峰值和吞吐；在显存目标以内把吞吐调到最高：先定混合精度等基础设置，再找每卡 batch，显存限制了吞吐时从轻到重加梯度检查点、梯度累积、ZeRO/FSDP 分片和 offload，不只靠调 batch；显存峰值接近可用显存、同时留出和用户商定的余量，利用充分又不 OOM；搭好断点续训的配置，并实际验证中断后能接着训练。只影响速度和显存的参数自行调整，会改变训练结果的参数先问用户；不写捕获 OOM 后自动缩小 batch 之类的静默兜底。 |
| [paper-explainer](paper-explainer/SKILL.md) | `/paper-explainer` | 任何领域的论文阅读与讲解：先定位整体问题和流程，说明组件角色、输入输出契约与基线瓶颈，再讲论文改动、新旧对比和具体例子，最后一句话总结。区分整体任务与单次接口，讲清依赖、并行、缓存复用和训练或评估信号。名词、公式、代码先解释作用；结论有原文依据，前后矛盾主动纠正。 |
| [simple-rules](simple-rules/SKILL.md) | `/simple-rules` | 面向人工表达式、约束和数值求解实现行为的算法，如控制、规划、调度和优化；普通深度学习训练不是主要场景。先理清流程与每条规则的作用，再与用户商议修改。规则精简、合理、有效、可验证；新增规则先解释用途、假设、影响与代价，经确认后才实施。避免叠加特例，不为减少数量擅删必要或安全约束；报告不堆代码名词。 |

## 安装

把需要的 skill 文件夹整个复制到 agent 的 skills 目录。以 clean-replace 为例：

```bash
# Qoder，所有项目可用
cp -r clean-replace ~/.qoder/skills/

# Qoder，只在当前项目可用
cp -r clean-replace <项目根目录>/.qoder/skills/

# Claude Code
cp -r clean-replace ~/.claude/skills/
```

复制时保留整个 skill 文件夹，包括其中的 `SKILL.md` 和配套文件。带有参考文件的 skill 会引用同目录下的其他文件，不能漏掉这些文件；只有 `SKILL.md` 的 skill 也按独立文件夹安装。

## 目录结构

```
Agent_Manager_Skills/
├── README.md               # 中文介绍
├── README.en.md            # 英文介绍
├── LICENSE
├── clean-replace/
│   ├── SKILL.md            # 主文件：触发条件、确认点、原则、执行流程、检查清单
│   ├── failure-modes.md    # 常见的失败方式
│   ├── rules.md            # 禁止事项、命名、测试、文件增长
│   ├── scanning.md         # 阶段 B：逐个文件扫描；阶段 C：清单格式
│   ├── change-index.md     # 修改索引（docs/agent-index/）
│   ├── artifacts.md        # 实验产物管理
│   ├── records.md          # 记录分工、哈希和镜像标签、临时文件、整理堆积的项目
│   ├── asking-user.md      # 什么时候问用户、怎么问
│   ├── verification.md     # 残留检查命令和最终报告格式
│   └── examples.md         # 完整示例
├── env-first/
│   └── SKILL.md            # 环境检查、“运行环境”格式、GPU/CUDA、Python 环境、下载排查、缓存位置、镜像标签、镜像源和代理、改文件、服务器
├── train-preflight/
│   ├── SKILL.md            # 主文件：触发条件、显存目标、原则、确认点、执行流程、最终报告、检查清单
│   ├── estimation.md       # 显存的组成、三种显存读数的区别、多卡、硬盘预算
│   ├── smoke-test.md       # 冒烟测试：跑通、测显存峰值、测吞吐、产物位置
│   ├── tuning.md           # batch size 和相关参数的调整顺序、学习率和 batch 的关系、经验来源
│   └── resume.md           # 断点续训：保存什么、怎么保存和恢复、怎么验证
├── paper-explainer/
│   └── SKILL.md            # 通用论文阅读与讲解：整体流程、接口、基线、改动、对比、例子
└── simple-rules/
    └── SKILL.md            # 规则算法设计哲学：先分析、解释和确认，再做最小有效改动
```

## 添加新的 skill

1. 在仓库根目录新建文件夹，文件夹名就是 skill 名，只用小写字母、数字和连字符。
2. 文件夹里必须有 `SKILL.md`，开头写 YAML frontmatter：

   ```markdown
   ---
   name: skill-name
   description: 做什么，什么时候用。用第三人称写，不超过 1024 字符。
   ---
   ```

3. `SKILL.md` 正文尽量控制在 500 行以内。内容多时拆成同目录下的参考文件，在 `SKILL.md` 里直接链接，不要再嵌套一层引用。
4. 需要固定执行的命令可以放进 `scripts/` 子目录。
5. 在上面的 Skill 列表里加一行。

## 许可证

[Apache License 2.0](LICENSE)
