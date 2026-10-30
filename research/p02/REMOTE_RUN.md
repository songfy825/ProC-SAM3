# P02 换机运行与 Codex 接手

此分支提供已有缓存的 CPU 离线动作空间分析和验收。换机后可以先验证工程契约；真实诊断仍需要已有真实缓存和独立来源证据。P02 GPU 缓存生产、生产账本和独立生产审计器是后续实现目标，当前没有可用的 P02 GPU 启动命令。仓库根目录的 P01 GPU 文档描述另一套流程，不能当作 P02 已完成的能力。

本机独立 clone 的迁移验收结果及源码/日志 hash 见 [migration/REPORT.md](migration/REPORT.md)。该证据不替代目标机器上的新验收。

## 1. 克隆正确分支

SSH（已配置 GitHub SSH 认证时）：

```bash
mkdir -p "$HOME/projects"
git clone --branch codex/p02-mask-action-space --single-branch \
  git@github.com:songfy825/ProC-SAM3.git "$HOME/projects/ProC-SAM3-p02"
cd "$HOME/projects/ProC-SAM3-p02"
git branch --show-current
git rev-parse HEAD
```

也可用 HTTPS 替换上面的 clone 命令；两种方式选一种，目标目录须尚不存在：

```bash
git clone --branch codex/p02-mask-action-space --single-branch \
  https://github.com/songfy825/ProC-SAM3.git "$HOME/projects/ProC-SAM3-p02"
```

目录和用户名可以不同；后续命令都从实际 checkout 根目录执行。不要使用 `--depth`：验收需要历史基点 `ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566`，会比较原 P01 文件是否保持冻结。

## 2. 选择 Python 环境

优先复用目标机已有的兼容环境。例如先执行 `source /absolute/path/to/env/bin/activate`，或 `conda activate <existing-env>`，再显式记录解释器：

```bash
export P02_PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

也可以直接写目标机现有解释器的绝对路径，例如 `export P02_PYTHON="$HOME/envs/proc-sam3/bin/python"`。不要复制原机器的用户名或路径，也不要假定系统 `python3` 已安装 torch。

`scripts/p02/python.sh` 的选择顺序为：

1. 显式 `P02_PYTHON`。
2. 已激活的 `VIRTUAL_ENV/bin/python`。
3. `CONDA_PREFIX/bin/python`。
4. 当前 checkout 的 `.venv/bin/python`。
5. 当前 checkout 的 `.p01-env/bin/python`。
6. PATH 中的 `python3`。

显式配置无效会报错，不会悄悄换解释器。包装脚本保留原 P01 对 Linux noexec 文件系统的 ELF loader 兼容路径；这不代表能补齐缺失的依赖。更换环境后重新执行 CPU 验收。

`requirements.txt` 来自已有 Ubuntu x86_64 / Python 3.12.14 环境，锁定 NumPy 1.26.4、torch 2.7.1+cu126、torchvision 0.22.1+cu126 等依赖。**全新环境安装尚未验证，本次迁移改造没有执行安装。** 如目标机没有兼容环境，可在已安装兼容 Python 3.12 的前提下选择下列独立 venv 路径；这是待在目标机验证的安装步骤：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
export P02_PYTHON="$(python -c 'import sys; print(sys.executable)')"
```

依赖安装会访问包索引；P02 分析和验收本身保持离线。若锁定版本无法安装，保留失败日志并查明原因，不要把改版本后的结果当成原环境复现。这里的 CPU 验收不要求运行 GPU 推理。

## 3. 配置运行目录并做 CPU 验收

```bash
export P02_RUNTIME="$HOME/runs/proc-sam3-p02"
P02_RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
bash scripts/p02/python.sh scripts/p02/analyze.py --help
bash scripts/p02/python.sh scripts/p02/verify.py --help
bash scripts/p02/python.sh scripts/p02/verify.py \
  --output "$P02_RUNTIME/audit/cpu-$P02_RUN_ID"
```

`P02_RUNTIME` 必须是绝对路径；开头的 `~` 会展开。未设置时，默认使用当前 checkout 的 `.p02-runtime`。所有 `--output` 必须是解析后运行根目录内尚不存在的子目录，不能等于运行根目录，也不能指向根目录外。相对运行根目录配置会报错。每次执行更新 `P02_RUN_ID`；失败目录也保留，不覆盖旧证据。

P02 覆盖继承的 `P01_PYTHON`、`P01_RUNTIME` 和 GPU 开关，设置 `P01_RUNTIME=<解析后的 P02_RUNTIME>/compat`、`P01_ENABLE_GPU=0`、空 `CUDA_VISIBLE_DEVICES` 及离线变量。Python 入口在可选库导入前安装 CPU/网络 guard。因此即使环境遗留 P01 GPU 配置，P02 也不会切换成 GPU 任务。

查看本次 `audit/cpu-<run-id>/verification.json` 和同目录日志。验收包含原 P01 CPU 契约、P02 测试、真实 CLI 的合成成功/损坏 receipt/缓存缺失/恢复检查，以及替换或删除 GT 后动作字节一致性检查。`PASS` 仅表示本次工程验收通过；合成输入仍标记 `REAL_ACTION_SPACE=PENDING`。历史 [REPORT.md](REPORT.md) 保留原执行偏差，不能用新验收抹去。

## 4. 已有真实缓存才能继续真实诊断

图像、GT、权重、真实缓存、完整大日志和 Python 环境不随 Git 克隆。将需要的已有资产另行放到目标机，保持文件内容及证据绑定关系；当前 CPU 分析不会重新读取权重字节，也不会生成缓存来补齐输入。

先阅读 [input.PROVENANCE.md](input.PROVENANCE.md)，再建立本机输入入口：

```bash
mkdir -p "$P02_RUNTIME/inputs"
cp -n research/p02/input.real.template.json "$P02_RUNTIME/inputs/real.json"
```

只用已有真实记录填写 `real.json` 中的 `path` 和原有独立记录的 SHA256。其引用可以是本机绝对路径，也可以是相对 `real.json` 所在目录的路径。development JSONL 中的相对图像路径相对该 JSONL；标签路径相对标签清单。来源契约中旧的 `R/runtime/p02` 示例在本机对应 `P02_RUNTIME`。

原生产 manifest/index/receipt/log、独立审计、代码清单、配置、use history 等带 hash 的源记录须保留原内容。不要批量替换其中的旧绝对路径再重算 hash，也不要重写历史报告和 receipt。若源记录内的旧路径在目标机不可访问，应记录具体迁移缺口并依据真实原始记录解决路径布局或适配问题；不能编造新的独立审计使检查通过。缺少合法 development/use history、所选 1008 缓存或独立来源证据时，保持 `PENDING`。

真实输入齐备后的命令：

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest "$P02_RUNTIME/inputs/real.json" \
  --output "$P02_RUNTIME/real-actions-$P02_RUN_ID"
```

可选 GT 必须使用独立 JSONL，并只传给评价入口：

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest "$P02_RUNTIME/inputs/real.json" \
  --labels "$P02_RUNTIME/inputs/labels.jsonl" \
  --output "$P02_RUNTIME/real-evaluated-$P02_RUN_ID"
```

标签每行包含 `dataset,image_id,path,sha256,image_sha256`，可选 `scene_id`。GT 不进入 development/cache 选择清单，不参与动作生成。缺少所选 GT 时评价为 `PENDING_GT`；损坏的已提供 GT 会失败。

尚无真实输入时，可运行以下命令确认缺证据路径；**预期退出码为 2**，不是一次真实实验成功：

```bash
bash scripts/p02/python.sh scripts/p02/analyze.py \
  --manifest research/p02/input.real.template.json \
  --output "$P02_RUNTIME/real-pending-$P02_RUN_ID"
```

| 退出码 | 含义 |
| --- | --- |
| `0` | 诊断或工程验收完成；合成成功、无 GT 完成均不等于真实科学结论成立。 |
| `1` | 已提供证据损坏/无效，或工程验收失败；检查日志与 `status.json`。 |
| `2` | 必需输入/缓存/来源证据缺失而 `PENDING`，或 CLI 用法错误；结合日志区分。 |

完整诊断产物说明见 [README.md](README.md)，冻结的动作、选择器和有界 oracle 规则见 [METHOD.md](METHOD.md)。有界 oracle 不是全局最优或上界；没有真实结果就不报告 mIoU 增益。

## 5. 交给另一台机器的 Codex

从 checkout 根目录打开 Codex，让它先读根目录 [AGENTS.md](../../AGENTS.md) 和本指南。可粘贴：

```text
请在当前 codex/p02-mask-action-space checkout 接手 P02。先读根 AGENTS.md，
再读 research/p02/ 下的 REMOTE_RUN.md、METHOD.md、input.PROVENANCE.md 和历史 REPORT.md。
确认分支、HEAD、工作区状态，以及本机 P02_PYTHON/P02_RUNTIME 的解析结果。
复用我指定的兼容环境，先运行 CPU verify 到全新的运行子目录，保存 receipt 和日志。
当前只运行已有 CPU analyze/verify；先报告真实输入是否齐备。
缺少缓存或来源证据保持 PENDING，勿把合成 PASS 当作真实实验通过。
不要改写历史 receipt，不要自动生成 GPU 缓存、下载模型或混入 GT 选样。
如需实现后续 P02 GPU producer/ledger/auditor，以我最新给定的目标和范围继续。
```

接手时以最新用户指令为准。历史报告中的绝对路径、提交状态和测试数字描述历史运行；当前代码状态用 `git` 核对，当前验收结果用新 receipt 核对。此指南不声称已完成目标机安装或真实 GPU 验证。
