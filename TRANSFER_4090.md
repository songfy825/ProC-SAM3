# Ubuntu x86_64 / RTX 4090 transfer

代码提供分阶段启动入口，**不是“只等 GPU 空闲即可保证实验成功”**。真实 GPU G0、1008/504 编码行为、显存容量、pilot 和 confirmation 均未实测；LoveDA 原始场景映射仍缺少可核验来源。补齐依赖、原始资产、场景证据和路径后，可以用一条命令从 G0 开始，失败即停止。没有后台排队或 GPU 空闲监控。

## 1. 代码和依赖

```bash
git clone --branch codex/p01-scale-calibration git@github.com:songfy825/ProC-SAM3.git
cd ProC-SAM3
conda create -n p01-sam3 -c conda-forge python=3.12.14 pip -y
conda activate p01-sam3
python -m pip install -r requirements.txt
python -m pip check
```

不要 shallow clone：验收会比对基点 `234c87e58fb08f02ba1cf163b2a73f5330d462e9`。依赖版本取自现有只读环境，包含 PyTorch `2.7.1+cu126` / torchvision `0.22.1+cu126`。本机 CPU 测试使用这些版本，**未在新 Ubuntu 环境执行安装**。目标机需要可用的 NVIDIA 驱动；驱动不由 requirements 安装。若包索引无法提供锁定版本，应保存安装失败日志并核查，勿静默换版本后复用旧 run_id。

`environment/` 保留可选的 Conda 元数据锁；按用户要求，不需要环境二进制包。`scripts/p01/pack_environment.py` 是此前的可选打包辅助工具，不属于 requirements 安装路径。模型代码由本仓库提供，无需另装 SAM3、新模型或 MLLM。

## 2. 准备只读资产和独立清单

权重、图像、GT、大缓存不在 Git 中。自行将已有资产放到目标机；代码不下载权重、不自动生成提示池。权重必须保持：

```text
sam3.pt: 3450062241 bytes
SHA256: 9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e
```

已发布提示池和词表在仓库内，其 hash 会校验。数据清单可由原工程 `manifests/LoveDA-inference.jsonl`、`iSAID-inference.jsonl` 迁移；清单内含图像 hash，无需重新选样或读取 GT。目标目录示例为 `/data/datasets/<LoveDA|iSAID>/img_dir/val/`：

```bash
mkdir -p /data/p01-inputs
bash scripts/p01/python.sh -m proc_research.prepare inventory \
  --manifest-dir /data/source-manifests --data-root /data/datasets \
  --output /data/p01-inputs/inventory-v1.jsonl
bash scripts/p01/python.sh -m proc_research.calibration.run split \
  --inventory /data/p01-inputs/inventory-v1.jsonl \
  --scene-map /data/p01-inputs/reviewed-scene-map.json \
  --limits '{"calibration":50,"pilot":50,"confirmation":80}' \
  --output /data/p01-inputs/splits-v1.jsonl
```

`reviewed-scene-map.json` 必须另行提供真实来源证据，格式见 [RUN_GPU.md](RUN_GPU.md)。文件名或非空说明不能证明场景独立性。清单跨集合图像 hash 和场景 ID 重叠必须为零；选样不能读取 GT。上面的 80 是容量示例，最终每集 calibration≤50、pilot≤50、confirmation≤200，总计≤600。先做无 GT 的容量估算并保留至少 20 GiB，规模调整生成新清单和新 run_prefix。

独立评价器接收 `pilot-labels.jsonl` 和 `confirmation-labels.jsonl`，每行仅 `dataset,image_id,path,sha256`。它们不能混入推理清单；confirmation 标签只在确认阶段评价时读取。已发布提示池构建来源不完整，不能声称已证明无泄漏。

## 3. 填写一次路径，先做 CPU 验收

```bash
cp research/p01/transfer.example.json /data/p01-inputs/4090-v1.json
# 编辑该 JSON：runtime、checkpoint、manifest、run_prefix 和两个独立标签清单路径。
export P01_RUNTIME=/data/p01-runs
bash scripts/p01/python.sh scripts/p01/verify.py --output "$P01_RUNTIME/audit/cpu-4090-v1"
bash scripts/p01/launch.sh preflight --settings /data/p01-inputs/4090-v1.json
bash scripts/p01/launch.sh pilot --settings /data/p01-inputs/4090-v1.json --dry-run
```

CPU 验收不构建模型；preflight 校验权重、提示池、所选图像 hash、场景隔离和容量；dry-run 仅打印计划。上述 output 不得复用。Conda 激活后无需设置 `P01_PYTHON`；若 shell 中已有该变量指向旧机器，应取消或改为目标环境的 Python。建议 `P01_RUNTIME` 与 JSON 的 runtime 相同，以便所有临时文件和日志集中存放。

## 4. 后续明确授权后运行 GPU（以下命令均尚未实测）

先独立冒烟：

```bash
P01_ENABLE_GPU=1 bash scripts/p01/launch.sh g0 \
  --settings /data/p01-inputs/4090-v1.json --allow-gpu
```

或者一条命令运行 G0 → calibration → fit → pilot → decode → 独立评价 → report：

```bash
P01_ENABLE_GPU=1 bash scripts/p01/launch.sh pilot \
  --settings /data/p01-inputs/4090-v1.json --allow-gpu
```

G0 必须满足：直接推理与缓存标签逐像素一致、分数最大绝对差≤1e-6、真实编码输入为 1008 和 504、特征尺寸有记录、缓存无损、单视图只编码一次。任一失败即停止，不扩大运行；缓存平均不涨点不会自动停止 ECD。详细验收及在线资源统计见 [RUN_GPU.md](RUN_GPU.md)。

查看 pilot 全部结果并冻结后，另行执行一次 confirmation：

```bash
P01_ENABLE_GPU=1 bash scripts/p01/launch.sh confirmation \
  --settings /data/p01-inputs/4090-v1.json --allow-gpu --pilot-reviewed
```

`--pilot-reviewed` 表示执行者已审核 pilot；入口不会替人作此判断。所有 11 种方法共享缓存；确认结果含场景配对 bootstrap 2000 次、95% 区间、种子 20261030。真实结果仍可能是负增益或证据不足。

## 5. 断点和产物

同一 settings 重跑时，已完成阶段先核验 receipt/hash、阶段身份及上游输入，再复用；未完成 GPU 阶段仅按相同身份恢复。fit/decode/evaluate/report 中断时默认停止；诊断后可给同一启动命令追加 `--retry-cpu`，将未完成目录连同文件 hash 留存在本次 `launcher-audits/.../incomplete-<stage>` 后重做该 CPU 阶段，已通过 GPU 工作不重复。已完成但 hash 损坏的结果不能用该选项绕过。配置、代码或清单改变必须创建新 run_prefix；模型失败不会自动重试或降精度。

产物：`runtime/runs/<run_prefix>-*`；逐阶段命令、退出码和日志：`runtime/launcher-audits/`；冻结输入：`runtime/inputs/`。保持 runtime 固定，迁移已运行实验时同时保留 `confirmation_claims/` 及原路径绑定；不得换根目录或删除声明来重复确认。当前未生成真实 GPU 运行数据。
