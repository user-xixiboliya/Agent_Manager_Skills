# 阶段 G：验证命令与最终报告格式

> clean-replace 的参考文件。阶段 G 开始前读。

以下检查全部通过，才能报告完成。

## G.1 搜索旧标识残留

```bash
git grep -n -i -E '<A.2 中列出、B 阶段补充后的所有旧标识，用 | 连接>'
```

要求：除以下白名单外，结果为零。

- CHANGELOG
- 实验记录文档
- 修改索引中的修改记录（`changes/` 目录）
- 实验产物目录
- 用户确认不修改的第三方代码
- 用户确认暂不修改的 AGENTS.md、CLAUDE.md 段落

文件职责表 `file-map.md` 描述的是当前状态，不在白名单中。

## G.2 检查版本化、备份和临时文件

```bash
git status --porcelain
git ls-files --others --exclude-standard
find . \
  -path ./.git -prune -o \
  -path ./node_modules -prune -o \
  -path ./.venv -prune -o \
  \( -iname '*_v[0-9]*' -o -iname '*_new*' -o -iname '*_old*' \
     -o -iname '*legacy*' -o -iname '*deprecated*' -o -iname '*_bak*' \
     -o -iname '*backup*' -o -iname '*.orig' -o -iname '*copy*' \
     -o -iname '*tmp*' -o -iname '*archive*' \) -print
```

要求：

- 没有本次新增的这类文件或目录。
- 项目原本就有的这类文件或目录，要在报告中列出，但不擅自处理。

## G.3 检查代码中的过渡标记

```bash
git diff | grep -n -i -E '^\+.*(legacy|deprecated|TODO.*remov|backward.?compat|use_old|旧逻辑|暂时保留)'
```

要求：本次新增的代码中没有这类标记。

## G.4 检查 fallback

```bash
git diff -U0 | grep -n -E '^\+.*(except\b|ImportError|\.get\(.*,|getattr\(.*,.*,|\bor\s+[A-Z]\w*\(|fallback|fall back|默认使用|退回)'
```

逐个检查命中项：

- 本次新增的 fallback 必须删除。
- 用户明确要求的 fallback 必须同时满足：
  - 触发时输出明确的日志或警告
  - 已经写进最终报告

另外，在整个仓库中检查是否还有退回旧实现的路径：

```bash
git grep -n -i -E '<旧标识>' -- '*.py' | grep -i -E 'except|get\(|or |default|fallback'
```

要求：结果为零。

## G.5 运行检查

运行项目已有的检查命令：

- 测试
- lint
- 类型检查

如果没有测试，至少完成：

- 所有被修改模块的 import 检查
- 一次最小可运行流程，例如 fast test 或 dry run

验证过程中产生的输出：

- 优先写到系统临时目录
- 必须写在项目中时，放进统一产物目录，并在报告中列出

## G.6 检查最终改动

查看：

```bash
git diff --stat
git status
```

确认：

- 每个新增文件都有必要
- 每个删除文件都在已确认清单中
- 没有修改与本次替换无关的文件
- 修改索引已更新，文件职责表中的哈希与当前文件一致

检查未通过时，回到对应阶段修复，然后重新执行全部检查。

## 最终报告格式

```text
## 替换内容
（A.2 中经用户确认的替换关系）

## 扫描覆盖
- 阅读数量、根据索引跳过的数量、交叉检查结果

## 清单完成情况
（确认点 2 中的清单，并标注每一行的完成状态）

## 修改文件
- path：修改内容

## 删除文件
- path：删除原因

## 新增文件
- path：新增原因；没有则写“无”

## fallback 检查
- 本次删除的 fallback：
- 保留的 fallback 及用户要求依据：没有则写“无”

## CHANGELOG 和实验记录
- 追加的条目：

## 修改索引更新
- 新增的修改记录：
- 更新的文件职责表行数：
- 修正的错误记录：

## 残留检查
- 旧标识搜索命令及结果：
- 版本化或备份文件检查结果：
- 过渡标记检查结果：
- 白名单及原因：

## 验证结果
- 执行的命令：
- 结果：

## 我做的判断
- 判断内容、理由、要修改时应改哪里；没有则写“无”

## 需要你处理的实验产物
| 路径 | 大小 | 属于 | 建议处理方式 |
|---|---|---|---|
（没有则写“无”）

## 建议，未执行
- 发现但不属于本次范围的问题；没有则写“无”

## 下一步需要你确认
- 是否提交 commit，并给出建议的 commit message
- 实验产物如何处理
```
