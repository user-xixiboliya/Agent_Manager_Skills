---
name: env-first
description: 在安装、下载、使用 GPU 或 CUDA 之前，先查清机器环境（系统、GPU 与驱动、CUDA、Python 环境、镜像源与代理、磁盘、服务器的工作空间、外网访问和共用数据目录），并写进项目的 AGENTS.md 或 CLAUDE.md；关于机器状态的结论必须有命令输出作为依据；下载失败时先确认目标是否存在、版本是否匹配，再查镜像源和代理；禁止不查现有环境就新建环境、用 sed 等命令改坏文件、擅自修改全局配置。遇到安装、下载、GPU 相关任务时自动启用，用户明确要求使用本 skill 时也启用。
---

# Env First：先搞清机器，再动手

## 什么时候启用

- 任务涉及安装或下载：Python 包、conda 包、apt 包、docker 镜像、CUDA toolkit、模型权重、数据集
- 任务涉及 GPU 或 CUDA：训练、推理、编译 CUDA 扩展、排查“没有 GPU”“CUDA 不可用”
- 用户明确要求使用本 skill

用户在本轮的明确要求与本规范冲突时，以用户要求为准。

---

## 七条规则

1. **结论要有依据**：“有没有 GPU”“装没装某个包”“网络通不通”，都要写出用的命令和关键输出。没查过的只能说“未检查”；命令本身失败了只能说“无法确认”。
2. **先找已有的，再考虑新建**：新建 Python 环境、安装 CUDA、下载模型之前，先查机器上是否已经有了。查过确实没有，再问用户。
3. **下载失败先查目标是否存在**：先确认包名、版本、tag 存在，并且和 Python 版本、系统版本、CPU 架构、CUDA 版本匹配，再查网络、镜像源和代理。
4. **不改全局配置**：pip、conda、docker、apt、git 的全局配置，`~/.bashrc`，`/usr/local/cuda` 软链接，都不由 agent 修改。需要临时换源或换路径时，只在单条命令上加参数或环境变量。
5. **用命令改文件前确认能恢复**：用 sed、重定向改文件前，确认文件在 Git 中且没有未提交的修改；改完立刻检查结果。
6. **失败后先查原因**：读完整报错，找第一个错误，针对原因处理。不原样重复同一条命令，不随手换版本、换源、换包。
7. **互相依赖的命令按顺序执行**：后一条命令依赖前一条的结果时（改脚本 → 运行脚本，commit → push，安装 → import），必须等前一条完成再发。

---

## 第一步：查清机器，写进 AGENTS.md 或 CLAUDE.md

### 先读已有的记录

1. 运行 `hostname`，确认当前在哪台机器上。
2. 读项目根目录的 AGENTS.md、CLAUDE.md，以及 README 中的环境说明。
3. 里面已经有“运行环境”一节时，直接使用，不再重新检查。只有执行中命令的实际输出和记录不一致时，以实际输出为准，并告诉用户哪里不一致、建议怎么改。
4. 没有“运行环境”一节，或者缺少本次任务需要的信息时，按下文检查。

### 检查

| 类别 | 命令 |
|---|---|
| 系统 | `cat /etc/os-release`、`uname -m`、`nproc`、`free -h`、`ls /.dockerenv`（判断是否在容器里） |
| 连接 | `echo $SSH_CONNECTION`、`who`、`sudo -n true && echo 有免密sudo` |
| GPU | `nvidia-smi`、`echo "${CUDA_VISIBLE_DEVICES-未设置}"` |
| CUDA | `which -a nvcc`、`nvcc -V`、`ls -d /usr/local/cuda*`、`readlink -f /usr/local/cuda`、`echo "${CUDA_HOME-未设置}"`、`gcc --version` |
| Python | `which -a python python3`、`conda env list`、`uv python list --only-installed`、`pyenv versions`、项目根目录 `ls -a` 看有没有 `.venv`、`uv.lock`、`environment.yml`、`requirements*.txt`、`pyproject.toml` |
| 镜像源和代理 | 见下文“镜像源和代理” |
| 磁盘 | `df -h`、`quota -s`、`du -sh ~/.cache/* 2>/dev/null` |
| 集群 | `sinfo`、`module avail` |
| 挂载和工作空间 | `df -h`、`mount \| grep -E ' /mnt\| /data\| nfs'`、`ls /mnt /data 2>/dev/null`、`pwd` |
| 外网 | `curl -sS -o /dev/null -m 8 -w '%{http_code}\n' https://pypi.org`，对 GitHub、HuggingFace、Docker Hub 同样测试 |

服务器上要特别查清下面几项：

| 项目 | 说明 |
|---|---|
| 工作空间在哪里 | 很多服务器的工作空间挂载在 `/mnt`、`/data` 下面，home 只有很小的空间。代码、环境、数据、输出都要放在工作空间里 |
| 能不能访问外网 | 服务器通常不能访问外网。不能访问时，不要反复尝试下载，按下文“服务器不能访问外网时”处理 |
| 共用数据集和权重放在哪里 | 很多服务器有存放共用数据集、预训练权重的目录。下载之前先在这些目录里找；这些目录只读，不修改、不删除、不往里写 |
| 内网镜像源或代理 | 不能访问外网的服务器，常有内网 pip 镜像、conda 镜像、docker 仓库或 HTTP 代理 |
| 是否和别人共用 | 决定能用哪些 GPU、能不能占满 CPU 和内存 |

命令查不出来的信息问用户：工作空间和共用数据目录的位置、能用哪几张 GPU、机器是否和别人共用、内网镜像源或代理的地址、项目用哪个 Python 环境（有多个候选时）。

### 写进 AGENTS.md 或 CLAUDE.md

检查结果写进项目根目录 AGENTS.md 或 CLAUDE.md 的“运行环境”一节，以后的会话直接读这一节。

- 项目里只有其中一个文件时，写进那个文件；两个都有时，问用户写哪个；都没有时，问用户要不要新建、建哪个。
- 修改这两个文件前，先把要写的内容给用户看，用户同意后再写。
- README 里已经写过的内容（项目介绍、安装步骤、用法），不要抄进来，需要时写一句“见 README 的某一节”。

“运行环境”一节的格式：

```markdown
## 运行环境

记录日期：YYYY-MM-DD

### 机器
- 运行位置：本机 / 服务器（SSH 别名 <别名>），是否和别人共用
- 系统：Ubuntu 22.04，x86_64
- GPU：8 × A100 80GB，驱动 535.161.08，驱动支持的最高 CUDA 12.2；本项目可用 2、3 号卡
- CUDA toolkit：12.1（/usr/local/cuda 指向此版本）、11.8；gcc 11.4

### 目录
- 工作空间：/mnt/<盘>/<用户名>/<项目>（home 空间小，不要往 home 写大文件）
- 共用数据集：/mnt/<盘>/datasets（只读）
- 共用预训练权重：/mnt/<盘>/pretrained（只读）
- 缓存：HF_HOME=/mnt/<盘>/<用户名>/hf_cache

### 网络
- 外网：不能访问
- 内网镜像源和代理：pip 用 <内网镜像地址>；HTTP 代理 <host>:<port>
- 需要外网资源时：在本机下载后用 rsync 传到 <目录>

### Python
- 解释器：/mnt/<盘>/<用户名>/miniconda3/envs/<名称>/bin/python
- 运行命令：`conda run -n <名称> --no-capture-output python train.py`
- 关键版本：torch 2.3.1+cu121

### 约定
- 不使用 sudo
```

不能写进去的内容（AGENTS.md 和 CLAUDE.md 常常提交到 Git 里）：

- 密码、token、API key、私钥，即使用户主动提供
- 代理地址中的用户名和密码：`http://user:pass@host:port` 只记 `http://host:port`
- 服务器 IP 和登录用户名：默认只记 SSH 别名和主机名，用户要求时才记

---

## GPU 和 CUDA

### 说“没有 GPU”之前

`nvidia-smi` 失败或看不到 GPU 时，原因可能是：

| 原因 | 检查方法 |
|---|---|
| `nvidia-smi` 不在 PATH 里 | `ls /usr/bin/nvidia-smi /usr/local/nvidia/bin 2>/dev/null`、`ls /dev/nvidia*` |
| 驱动没加载，常见于内核升级后 | 报错含 “couldn't communicate with the NVIDIA driver”；`lsmod \| grep nvidia` |
| 容器启动时没加 `--gpus` | `ls /.dockerenv`、`ls /dev/nvidia*` |
| `CUDA_VISIBLE_DEVICES` 为空或为 `-1` | `echo "${CUDA_VISIBLE_DEVICES-未设置}"` |
| 命令运行在 agent 的沙箱里，看不到设备 | 报错带 `bwrap` 或权限错误；申请在沙箱外运行 |
| 查的是本机，GPU 在服务器上 | `hostname` |
| 不是 NVIDIA 显卡 | `lspci \| grep -i -E 'vga\|3d'` |

逐项排除后才能说“没有 GPU”，否则只能说“无法确认”，并写出已运行的命令。

`torch.cuda.is_available()` 返回 False 也不等于没有 GPU：可能装的是 CPU 版 torch（`torch.version.cuda` 为 None），可能 torch 的 CUDA 版本高于驱动支持的版本，也可能 `CUDA_VISIBLE_DEVICES` 把卡屏蔽了。

### 四个 CUDA 版本

| 版本 | 查看方式 | 含义 |
|---|---|---|
| 驱动支持的最高 CUDA | `nvidia-smi` 右上角 | 上限。不代表装了 toolkit |
| toolkit | `nvcc -V` | 编译 CUDA 扩展时使用。可能装了多个 |
| torch 自带的 CUDA | `python -c "import torch; print(torch.version.cuda)"` | pip 版 torch 自带运行库，不依赖本机 toolkit |
| cuDNN | `python -c "import torch; print(torch.backends.cudnn.version())"` | |

- torch 的 CUDA 版本不能高于驱动支持的最高版本。
- 编译 CUDA 扩展（flash-attn、apex、自定义算子等）时，`nvcc` 的版本要和 `torch.version.cuda` 一致，gcc 版本要在 toolkit 支持的范围内。
- 有多个 toolkit 时，用 `CUDA_HOME=/usr/local/cuda-12.1 PATH=/usr/local/cuda-12.1/bin:$PATH <命令>` 指定，不改 `/usr/local/cuda` 软链接。
- `nvidia/cuda:<版本>` 镜像要求宿主机驱动支持该版本。
- 安装驱动或 CUDA toolkit 前必须问用户。

### 使用 GPU 前

- 用 `nvidia-smi` 看各卡的显存和利用率，选空闲的卡，并遵守用户约定。
- 用 `CUDA_VISIBLE_DEVICES` 明确指定卡号。
- 显存不足时，先看是不是别人占着，再考虑减小 batch size。
- 不结束不是自己启动的 GPU 进程。

---

## Python 环境

### 先找现有环境

1. 项目文档和脚本中写明的环境：`grep -rn -E "conda activate|source .*activate|uv run|poetry run" --include=*.sh --include=*.md .`
2. 项目根目录的 `.venv/`、`venv/`；有 `uv.lock` 说明用 uv 管理。
3. conda 环境：`conda env list`。conda 不在 PATH 里时，看 `~/miniconda3/envs`、`~/anaconda3/envs`、`~/.conda/envs`。
4. 名字和项目相关的环境，用 `<解释器> -c "import torch; print(torch.__version__)"` 确认关键包能导入。
5. 有多个候选或都不合适时，列出各候选的路径、Python 版本、关键包，问用户。

只有查过没有合适的环境、并且用户同意后，才新建。不新建 `env2`、`myenv_new` 这类重复环境，不删除已有环境。

### 运行命令

- 每次执行命令可能都是一个新的 shell，上一次的 `conda activate` 不会保留。用解释器的完整路径、`conda run -n <名称> --no-capture-output python ...`，或者 `uv run python ...`。
- 装包一律用 `<解释器> -m pip`，不单独用 `pip`，避免装到别的环境里。

### 安装包

- 安装前先查是否已经装了：`<解释器> -m pip show <包名>`。
- 先用 `--dry-run` 看会改动哪些包。会改动 torch、numpy 大版本、`nvidia-*` 库时，先问用户。
- 系统 Python 报 “externally-managed-environment” 时，不加 `--break-system-packages`，不用 `sudo pip`，改用项目环境。
- 安装名和导入名可能不同：`opencv-python` 对应 `cv2`，`scikit-learn` 对应 `sklearn`，`PyYAML` 对应 `yaml`，`pillow` 对应 `PIL`。

---

## 下载失败的排查顺序

1. **读完整报错**，判断类型：

   | 报错 | 通常原因 |
   |---|---|
   | `No matching distribution found`、`Could not find a version` | 包名错、版本不存在、当前 Python 版本或平台没有 wheel、镜像缺包 |
   | docker `manifest unknown`、`not found` | tag 不存在 |
   | `404` | URL 或版本号写错 |
   | apt `Unable to locate package` | 当前发行版里包名不同，或者没运行 `apt update` |
   | conda `PackagesNotFoundError` | 当前 channel 里没有 |
   | HuggingFace `Repository Not Found` | 仓库名错，或者需要登录授权 |
   | `Connection timed out`、`Could not resolve host` | 网络、DNS、代理 |
   | `SSL: CERTIFICATE_VERIFY_FAILED` | 代理替换了证书、系统时间不对、证书过旧 |
   | `401`、`403` | 需要登录或授权，请用户自己登录 |

2. **确认目标存在，版本匹配**：

   | 来源 | 命令 |
   |---|---|
   | PyPI | `<解释器> -m pip index versions <包名>`；`curl -s https://pypi.org/pypi/<包名>/json` 看版本和 `requires_python` |
   | PyTorch | 到官方历史版本页面查 torch、torchvision、torchaudio 的对应版本；`curl -s https://download.pytorch.org/whl/cu121/torch/` 看有没有对应 Python 版本和平台的 wheel |
   | docker | `docker manifest inspect <镜像>:<tag>` |
   | apt | `apt-cache policy <包名>`、`apt-cache search <关键词>` |
   | conda | `conda search -c conda-forge <包名>` |
   | HuggingFace | `curl -sI <HF 地址>/<仓库>/resolve/main/config.json` |
   | URL 下载 | `curl -sI <url>` |

   要对上的版本：Python 版本、系统版本（ubuntu2204 和 ubuntu2404 的安装包不同）、CPU 架构（x86_64、aarch64）、CUDA、torch 和 torchvision 的对应关系、编译扩展和 torch 的对应关系、numpy 1 和 numpy 2。版本号要来自官方兼容表或包索引，不凭记忆写，并在回复中写明来源。

3. **确认这次下载用的是哪个源、有没有走代理**，见下文“镜像源和代理”。
4. **分别测试官方源和镜像源**：`curl -sS -o /dev/null -m 8 -w '%{http_code} %{time_total}s\n' <地址>`。
5. **定位到原因后再重试**。只有偶发的网络超时可以原样重试一次。

不要做的事：随手换镜像源试试；加 `--trusted-host`、`curl -k` 跳过 SSL 校验；悄悄降版本或换成另一个包。这些都要先问用户。

### 服务器不能访问外网时

1. 先在共用数据集、共用权重目录里找。
2. 用“运行环境”中记录的内网镜像源或代理。
3. 都没有时，在能联网的机器（通常是本机）上下载，再传到服务器的工作空间：

   | 内容 | 本机 | 服务器 |
   |---|---|---|
   | 文件、数据集、权重 | 下载后 `rsync -avP <本地路径> <别名>:<工作空间>/` | — |
   | pip 包 | `pip download -d wheels <包名> --python-version <服务器 Python 版本> --platform manylinux2014_x86_64 --only-binary=:all:` | `<解释器> -m pip install --no-index --find-links wheels <包名>` |
   | docker 镜像 | `docker save -o img.tar <镜像>` | `docker load -i img.tar` |
   | HuggingFace 模型 | 下载到本地目录后 rsync | 用本地路径加载，或设置 `HF_HUB_OFFLINE=1` |

   在本机下载时，Python 版本、平台、CUDA 版本要按服务器的情况选，不能按本机的。
4. 传输前告诉用户要传什么、多大、传到哪里。

大文件：下载前先看大小和目标磁盘剩余空间，放到“运行环境”中记录的工作空间或数据盘。超过 10 GB 或空间不够时先问用户。home 空间小时，用 `HF_HOME`、`TORCH_HOME`、`PIP_CACHE_DIR` 把缓存指到数据盘。

---

## 镜像源和代理

每个工具的配置是分开的，pip 配好了不代表 docker 和 git 也能用。

| 工具 | 镜像源 | 代理 | 查看命令 |
|---|---|---|---|
| pip | `pip.conf`、`PIP_INDEX_URL` | 环境变量、pip.conf | `pip config list` |
| uv | `uv.toml`、`pyproject.toml` 的 `[tool.uv]`、`UV_INDEX_URL` | 环境变量 | `cat ~/.config/uv/uv.toml` |
| conda | `.condarc` 的 channels | `.condarc` 的 `proxy_servers` | `conda config --show channels proxy_servers` |
| docker | `/etc/docker/daemon.json` 的 `registry-mirrors` | systemd 配置，不读 shell 环境变量 | `docker info \| grep -i -A3 -E 'mirror\|proxy'` |
| apt | `/etc/apt/sources.list`、`sources.list.d/` | `/etc/apt/apt.conf.d/` | `grep -r -i proxy /etc/apt/apt.conf.d/` |
| HuggingFace | `HF_ENDPOINT` | 环境变量 | `env \| grep -i '^hf_'` |
| git | — | `http.proxy` | `git config --get-regexp proxy` |
| 通用 | — | `http_proxy`、`https_proxy`、`all_proxy`、`no_proxy`，大小写都要查 | `env \| grep -i proxy` |

常见的坑：

- 镜像同步有延迟，刚发布的版本镜像里可能还没有。
- docker 的 `registry-mirrors` 只对 Docker Hub 生效，`nvcr.io`、`ghcr.io` 不经过它。
- 国内 pip 镜像通常没有 torch 的 CUDA 版本，要从 `download.pytorch.org` 下载。
- `--index-url` 会替换原来的源，其他依赖跟着找不到；装 torch 时用 `--extra-index-url`，或者分两步装。
- `sudo` 默认不保留代理环境变量。
- `no_proxy` 没配好时，访问 localhost 和内网地址也走代理。
- 配置里的镜像可能已经停止服务。

临时换源的写法：

```bash
<解释器> -m pip install -i <镜像地址> <包名>
<解释器> -m pip install torch --extra-index-url https://download.pytorch.org/whl/cu121
conda install -c conda-forge --override-channels <包名>
HF_ENDPOINT=<镜像地址> huggingface-cli download <仓库>
git -c http.proxy=http://<host>:<port> clone <url>
```

---

## 用命令改文件

| 写法 | 后果 | 改用 |
|---|---|---|
| `sed -i` 正则没先测 | 整个文件被替换错 | 先 `sed '<表达式>' f \| diff f -` 预览 |
| `sed` 改多行，或内容含 `/`、`&`、`\` | 匹配错或转义错 | 编辑工具 |
| `find ... -exec sed -i` 批量改 | 一次改坏很多文件 | 先列出文件，抽一个预览 |
| `cmd f > f` 读写同一个文件 | 文件被清空 | 编辑工具，或写临时文件再 `mv` |
| 该用 `>>` 时用了 `>` | 原内容被覆盖 | 写之前确认 |
| heredoc 写成 `<<EOF` | `$变量` 和反引号被展开 | `<<'EOF'` |
| 为了恢复，运行 `git checkout -- .`、`git reset --hard`、`git clean -fd` | 用户未提交的修改全部丢失 | 只恢复自己改坏的那个文件，先问用户 |

- 改文件优先用编辑工具。
- 用命令改之前看 `git status`。文件没被 Git 跟踪，或者有用户未提交的修改时，先复制到系统临时目录（`/tmp`），不要在项目里留备份文件。
- 改完立刻看 `git diff`，确认文件没有变空；能做语法检查的就做（`python -m py_compile`、`python -m json.tool`）。

---

## 服务器和共用机器

- 每条关键命令都要清楚在哪台机器上执行。本机的路径不能拿去服务器上用。
- 非交互的 SSH 命令（`ssh <别名> '<命令>'`）通常不加载 `.bashrc`，conda 和 PATH 可能都不一样，用解释器的完整路径。
- 共用机器上不做：`pkill python`、`killall`、结束别人的进程、`sudo`、`chmod -R 777`、`docker system prune`、删除别人的容器、清理共用缓存。
- 有 slurm 的集群，计算任务通过 `srun`、`sbatch` 提交，不在登录节点上运行。
- 要下载数据集或权重之前，先在“运行环境”中记录的共用目录里找。共用目录只读。
- 预计超过几分钟的任务放到后台：`tmux new -d -s <名称> '<命令> 2>&1 | tee <日志>'`，或者 `nohup <命令> > <日志> 2>&1 &`，记下会话名或 PID。启动前先确认同样的任务没有在运行，端口没被占用（`ss -ltnp | grep :<端口>`）。

---

## 必须先问用户

- 新建 Python 环境
- 安装会改动 torch、numpy 大版本、`nvidia-*` 库的包
- 安装驱动、CUDA toolkit、系统包，或者使用 sudo
- 修改全局配置
- 结束不是自己启动的进程，或者使用用户没允许的 GPU
- 下载超过 10 GB 的文件，或者目标磁盘空间不够
- 清理缓存、删除环境、镜像或容器
- 跳过 SSL 校验、降低版本、换成另一个包
- 修改或新建 AGENTS.md、CLAUDE.md
- 机器状态无法确认

提问时先写已经查过的内容和关键输出，再给 2–4 个选项，标出推荐的选项和理由。

---

## 回复中写明

- 在哪台机器上执行，用的哪个解释器（完整路径）
- 涉及的关键版本：torch、CUDA、驱动
- 本次对环境的改动：装了哪些包、下载了什么文件到哪里、启动了哪些后台任务以及如何查看和停止
- AGENTS.md 或 CLAUDE.md 的“运行环境”更新了哪些内容，或者建议更新哪些内容
