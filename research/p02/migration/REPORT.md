# P02 路径配置与迁移验收

本轮迁移工程完成，真实动作空间、oracle 和效果仍为 PENDING。实现提交 `49e523704273a73e844b4b934ce42c52a24c3414` 已推送至 `songfy825/ProC-SAM3` 的 `codex/p02-mask-action-space` 分支，并以 `git ls-remote` 核对同一 SHA。本报告由后续纯文档提交收录；当前分支 HEAD 以 Git 为准。

## 改动与范围

- `P02_PYTHON` 选择本机解释器；优先级为显式路径、激活的 virtualenv、Conda、checkout `.venv`、`.p01-env`、PATH `python3`。无效显式值直接失败。
- `P02_RUNTIME` 接受绝对路径或 `~`，默认 checkout 下 `.p02-runtime`。输出须为解析后的根目录内全新子目录；相对根目录、越界、符号链接逃逸、覆盖已有结果均拒绝。
- 保留 CPU/离线/网络禁用规则；覆盖继承的 P01 解释器、运行目录与 GPU 设置。算法、METHOD、分析/验收 CLI 和原 P01/model 文件不变。旧测试只替换机器路径假设。
- 新增 16 项迁移测试；根 `AGENTS.md`、`REMOTE_RUN.md` 和 README 提供另一台机器与 Codex 接手步骤。

## 验证

| 场景 | P01 | P02 | 跳过 | 完整验收 |
| --- | ---: | ---: | ---: | --- |
| 原 checkout 最终代码，source-v2 | 60 | 78 | 0 | PASS |
| 独立 clone，代码/运行目录均含空格 | 60 | 78 | 0 | PASS |

78 项 P02 = 原 62 项 + 新 16 项。两个验收都核对 CLI 成功退出 0、损坏 receipt 退出 1、缺缓存/来源或错误参数退出 2、恢复成功，以及删除/替换 GT 后动作逐字节一致。真实缺证据检查保持 PENDING。

独立 clone 使用 `--no-hardlinks`，检查 197 个对应对象文件不是同一 inode、无 alternates，验收前后工作区干净。故意继承 `P01_ENABLE_GPU=1`、`CUDA_VISIBLE_DEVICES=0`、无效 P01 解释器及独立旧运行目录，P02 仍覆盖为 CPU 配置，旧运行目录未创建。两次 receipt 的源码 hash 均与实现提交一致。

独立审查发现 `P02_RUNTIME='~/...'` 与带 `~` 的输出路径展开不一致；先用测试复现，再修复并通过完整验收。符号链接测试使用临时 `/tmp` fixture，因为共享文件系统创建 symlink 返回 ENOSYS；原断言保留，没有 skip。源码/日志位于 [run_manifest.json](run_manifest.json) 及 [evidence](evidence)，JSON receipt 保留原字节，文本转录仅去除行尾空白，原日志保留在本机 `runtime/p02/migration`。

## 实际命令

原目录最终验收：

```bash
cd /home/fengyi/CodeHome/Proc/src/ProC-SAM3-p02
P02_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python \
P02_RUNTIME=/home/fengyi/CodeHome/Proc/runtime/p02/migration \
bash scripts/p02/python.sh scripts/p02/verify.py \
  --output /home/fengyi/CodeHome/Proc/runtime/p02/migration/audit/source-v2
```

独立 clone 与完整迁移验收：

```bash
git clone --no-hardlinks --single-branch --branch codex/p02-mask-action-space \
  /home/fengyi/CodeHome/Proc/src/ProC-SAM3-p02 \
  '/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated checkout with spaces'
cd '/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated checkout with spaces'
P02_PYTHON=/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python \
P02_RUNTIME='/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated runtime with spaces' \
P01_RUNTIME='/home/fengyi/CodeHome/Proc/runtime/p02/migration/forbidden inherited runtime' \
P01_PYTHON=/must-not-inherit/python P01_ENABLE_GPU=1 CUDA_VISIBLE_DEVICES=0 \
bash '/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated checkout with spaces/scripts/p02/python.sh' \
  '/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated checkout with spaces/scripts/p02/verify.py' \
  --output '/home/fengyi/CodeHome/Proc/runtime/p02/migration/relocated runtime with spaces/audit/full-v1'
```

发布使用单独的 `fork` remote，保留指向 C 的本地 `origin`：`git push -u fork HEAD:refs/heads/codex/p02-mask-action-space`。首次受限网络 DNS 失败，获准在联网执行环境重试成功。没有 force push、merge 或 PR。

## 限制与历史

本次是同一 Linux 主机、已有 Python 3.12.14 / torch 2.7.1+cu126 环境的路径迁移验证。未安装新依赖，未验证另一台物理机器；数据、GT、权重、缓存和环境不随 Git 搬迁。后续机器应按 [REMOTE_RUN.md](../REMOTE_RUN.md) 先验收。

本次未执行 GPU 探测、推理、MLLM/API 或模型下载。此前运行的 CUDA availability-query 偏差仍保留在历史 REPORT，不被本次 PASS 抹去。P02 GPU producer、生产账本和独立生产 auditor 尚未实现；真实输入不足仍是科学结果的阻断项。

C 在本轮被其他工作从 `e8e07cb` 推进至 `4013778`，本任务只读检查并记录，没有修改 C，也没有纳入其新提交。P02 继续冻结原 `ec7f57e` 基点的 P01/model 契约。
