# Agent_Manager_Skills

这里收集一些用来约束编程 agent 工作方式的 skill。每个 skill 放在单独的文件夹里，可以单独安装。

## Skill 列表

| Skill | 调用方式 | 用途 |
|---|---|---|
| [clean-replace](clean-replace/SKILL.md) | `/clean-replace` | 替换方法、模型、接口或实现路线时，彻底删除旧实现，同步更新所有引用、配置、测试和文档。禁止版本化副本、备份文件、兼容层和静默 fallback。修改前逐个文件扫描项目，并维护修改索引来缩小扫描范围。 |

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

复制时保留整个文件夹。`SKILL.md` 会引用同目录下的其他文件，只复制 `SKILL.md` 不能正常使用。

## 目录结构

```
Agent_Manager_Skills/
├── README.md
├── LICENSE
└── clean-replace/
    ├── SKILL.md            # 主文件：触发条件、确认点、原则、执行流程、检查清单
    ├── failure-modes.md    # 常见的失败方式
    ├── rules.md            # 禁止事项、命名、测试、文件增长
    ├── scanning.md         # 阶段 B：逐个文件扫描；阶段 C：清单格式
    ├── change-index.md     # 修改索引（docs/agent-index/）
    ├── artifacts.md        # 实验产物管理
    ├── asking-user.md      # 什么时候问用户、怎么问
    ├── verification.md     # 残留检查命令和最终报告格式
    └── examples.md         # 完整示例
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
