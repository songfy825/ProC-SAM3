# P01 数据和权重迁移包

本包保存已有资产的原始字节，不包含新模型或训练结果。现有实验代码、requirements 和配置文件保持不变。解压后用软链接兼容原机绝对路径与已发布配置中的 `/data` 路径，数据只需保存一份。

## 包含什么

| 内容 | 本机原位置 | 数量 / 原始字节 |
|---|---|---|
| SAM3 权重 | `/home/fengyi/sam3.pt` | 1 / 3,450,062,241 |
| LoveDA 验证图像 | `/home/fengyi/E/data/SegEarth/LoveDA/img_dir/val/` | 1,669 / 2,411,772,953 |
| LoveDA 标签 | `/home/fengyi/E/data/SegEarth/LoveDA/ann_dir/val/` | 1,669 / 15,695,837 |
| iSAID 验证图像 | `/home/fengyi/E/data/SegEarth/iSAID/img_dir/val/` | 11,644 / 28,044,597,288 |
| iSAID 标签 | `/home/fengyi/E/data/SegEarth/iSAID/ann_dir/val/` | 11,644 / 37,047,487 |
| 原始清单 | `/home/fengyi/CodeHome/ProC-SAM3-repro/manifests/` | 仅 LoveDA/iSAID 的推理清单和配对资产清单 |
| 已发布提示池、类别词表、BPE | Git 仓库 `prompt_pools/`、`configs/`、`sam3/assets/` | 仓库已有，包内留 hash 可核验副本 |

未包括其他五个数据集、训练集、旧实验预测、环境二进制包。完整验证池只用于迁移候选资产；P01 仍限每集 calibration≤50、pilot≤50、confirmation≤200、合计≤600。标签文件不重编码，iSAID 标签后缀也保持原样。GT 只由独立评价器解释；迁移过程仅复制并校验字节。

SAM3 SHA256：`9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。

## 1. 传输和校验

把 `p01-assets.tar.gz`、`SHA256SUMS`、`verification.json` 一起传到目标机。同目录执行：

```bash
sha256sum -c SHA256SUMS
mkdir -p "$HOME/p01-transfer"
tar -xzf p01-assets.tar.gz -C "$HOME/p01-transfer" --no-same-owner
```

解压目录为 `$HOME/p01-transfer/p01-assets`，也可选其他有空间的磁盘。约 31.7 GiB 解压空间，另外为后续 FP32 缓存预留容量和至少 20 GiB 空闲。具体推理容量由 preflight 按最终清单估算。归档内 `MANIFEST.json` 保存每个文件的原位置、字节数和 SHA256。

如需逐文件重新验证整个归档（只用 Python≥3.10 标准库，无 GPU）：

```bash
python "$HOME/p01-transfer/p01-assets/package.py" verify /实际路径/p01-assets.tar.gz
```

## 2. 代码和环境

使用已推送的 `codex/p01-scale-calibration` 分支。包对应生产代码 SHA256 `ca1816ca8606689c2f4077851d89819758b38dead540c6be86807563659eb9a3`，初始可用提交 `ec7f57ef2a3ec5cb166149a2fa85f5eff4a2f566`；后续迁移辅助脚本和记录不修改这份实验代码。

```bash
git clone -b codex/p01-scale-calibration git@github.com:songfy825/ProC-SAM3.git
cd ProC-SAM3
conda create -n p01-sam3 -c conda-forge python=3.12.14 pip -y
conda activate p01-sam3
python -m pip install -r requirements.txt
```

新机安装和真实 GPU 仍未验证。代码路径可任意，下一步会链接到旧路径。不要移动已经完成实验的 runtime 或删除 confirmation_claims。

## 3. 一次建立路径兼容，不编辑代码或配置

仍在克隆仓库目录中，先预览：

```bash
python "$HOME/p01-transfer/p01-assets/setup_links.py" --code "$PWD"
```

确认输出的源和目标路径后执行。创建 `/data` 与 `/home/fengyi` 下路径通常需要管理员权限；仅这一步可能需要 sudo。脚本不会执行模型、安装软件或启动实验：

```bash
sudo "$(command -v python)" "$HOME/p01-transfer/p01-assets/setup_links.py" --code "$PWD" --apply
```

已有路径指向其他文件或目录时，脚本整体拒绝执行，不删除、不覆盖。需先人工核对冲突；不要直接 `rm -rf`。相同链接可重复检查。不要在原机执行 `--apply`，以免与原始资产路径冲突。

关键对应关系：`/home/fengyi/sam3.pt` 与 `/data/weights/sam3.pt` 指向同一权重；原 `/home/fengyi/E/data/SegEarth/{LoveDA,iSAID}` 与 `/data/datasets/{LoveDA,iSAID}` 指向同一数据；`/data/p01-inputs` 指向包的 inputs；`/data/p01-runs` 指向包的 runtime；原代码路径指向你的 Git 克隆目录。

包中 `inputs/4090-v1.json` 是仓库 `research/p01/transfer.example.json` 的逐字节副本，内容不需修改。不要 sudo 运行后续数据准备或实验。

## 4. 仍需真实场景来源证据，不能用软链接解决

**包内没有**已审核的 `reviewed-scene-map.json`、最终 splits、pilot-labels 或 confirmation-labels。这些从未存在，本次不伪造。LoveDA 数字文件名不能证明原始场景独立，iSAID parent image 名称也不证明地理独立。

提供经审核的场景映射后，以下 CPU 命令会生成外部输入清单，配置文件本身保持不变：

```bash
bash "$HOME/p01-transfer/p01-assets/prepare_inputs.sh" /实际位置/reviewed-scene-map.json
```

映射格式见代码仓库 RUN_GPU.md；默认每集 50/50/80，是冻结上限内的容量示例。选样不读取 GT；仅在划分完成后，从独立标签元数据中筛出对应评价清单。输入清单已有时会停止，不能覆盖旧实验。预估容量不足则停止，不自动降精度、删方法或扩大规模。

## 5. 后续 GPU 命令保持原样（尚未实测）

在目标机、明确决定开始后：

```bash
cd /home/fengyi/CodeHome/Proc/src/ProC-SAM3
export P01_RUNTIME=/data/p01-runs
P01_ENABLE_GPU=1 bash scripts/p01/launch.sh pilot --settings /data/p01-inputs/4090-v1.json --allow-gpu
```

该入口先运行真实 G0，失败即停；通过才继续 calibration 和有界 pilot。confirmation 仍须审核 pilot 后另行执行。原始 G0 阈值、场景隔离、bootstrap 与资源报告要求全部保留。当前没有 GPU 推理、空闲等待或队列，研究结论仍为 PENDING。
