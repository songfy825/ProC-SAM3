# 192.168.0.18：仅迁移权重，复用现有数据

2026-10-30，只读逐文件 SHA256 比对结果：本机与 `fengyi@192.168.0.18` 的 LoveDA/iSAID **验证图像和标签完全一致**，共 26,626 文件、30,509,113,565 字节；缺失、额外文件、大小不符、hash 不符均为 0。两边实际清单摘要一致：`a8eb43b4735e2d1094eb422efaf9e6f327ccb0c80b4fbc96e9fef49de4470011`。

| 数据 | 本机 | 目标机 | 图像 + 标签 |
|---|---|---|---|
| LoveDA | `/home/fengyi/E/data/SegEarth/LoveDA` | `/home/fengyi/data/SegEarth/loveDA` | 1,669 + 1,669 |
| iSAID | `/home/fengyi/E/data/SegEarth/iSAID` | `/home/fengyi/data/SegEarth/iSAID` | 11,644 + 11,644 |

比对仅覆盖 `img_dir/val` 和 `ann_dir/val`，不宣称其他数据集或训练集一致。`LoveDA` / `loveDA` 大小写差异可用软链接解决，**不改图像和标签、不改现有实验代码和配置**。

目标 `/home/fengyi/sam3.pt` 不存在。唯一需要迁移的大文件是本机 `/home/fengyi/sam3.pt`：3,450,062,241 字节（约 3.21 GiB），SHA256 `9999e2341ceef5e136daa386eecb55cb414446a00ac2b55eb2dfd2f7c3cf8c9e`。

本次只读远端；未上传、创建软链接、改动文件或运行 GPU。以下操作供之后实际迁移时执行。完整数据归档已取消，其不完整文件已清理，只保留取消记录和日志。不要传输旧的 `assets-v1` 目录。

## 交付内容

本机 `runtime/p01/transfer/weight-only-v1/`：

- `sam3.pt`：已核验的权重副本，原权重保留。
- `p01-metadata-v2.tar.gz`：少量原清单、标签文件名/hash 元数据、已发布提示/词表副本、路径辅助脚本及未改动的 `4090-v1.json`；不含任何数据图像、标签图像或环境二进制包。
- `SHA256SUMS`、`verification.json`：迁移校验。
- `REMOTE_TRANSFER.md`：本说明。

小元数据包是为了保持现有清单和路径兼容；数据本体无需传输。

## 迁移（本次未执行远端写入）

在本机交付目录中：

```bash
sha256sum -c SHA256SUMS
scp sam3.pt p01-metadata-v2.tar.gz SHA256SUMS fengyi@192.168.0.18:/home/fengyi/
```

目标 `/home/fengyi/sam3.pt` 本次核验时不存在；执行前若已存在，应先检查 hash，不能覆盖未知文件。SSH 密码不写入脚本、日志或命令行参数。

在目标机 `/home/fengyi`：

```bash
sha256sum -c SHA256SUMS
mkdir -p /home/fengyi/p01-transfer
tar -xzf p01-metadata-v2.tar.gz -C /home/fengyi/p01-transfer --no-same-owner
```

代码从已推送分支克隆，Conda 依赖仍使用仓库 requirements.txt。目标系统 Python 为 3.8；路径辅助脚本使用已激活的 **Conda Python 3.12**。不要用系统 Python 执行路径辅助脚本。

## 维持原路径与配置

激活 p01-sam3 环境后，在代码克隆目录预览：

```bash
python /home/fengyi/p01-transfer/p01-assets/setup_links.py \
  --code "$PWD" --existing-data-root /home/fengyi/data/SegEarth \
  --loveda-name loveDA --checkpoint /home/fengyi/sam3.pt
```

确认预览无冲突后，才创建路径别名（已有其他内容时整体拒绝，不覆盖）：

```bash
sudo "$(command -v python)" /home/fengyi/p01-transfer/p01-assets/setup_links.py \
  --code "$PWD" --existing-data-root /home/fengyi/data/SegEarth \
  --loveda-name loveDA --checkpoint /home/fengyi/sam3.pt --apply
```

这会使旧 `/home/fengyi/E/data/SegEarth/LoveDA` 和 `/data/datasets/LoveDA` 都指向目标已有的 `loveDA`；iSAID 同理。`/data/weights/sam3.pt` 指向 `/home/fengyi/sam3.pt`。`/data/p01-inputs`、`/data/p01-runs` 指向元数据包的 inputs、runtime。原代码路径链接到实际克隆目录。配置文件无需编辑。

## 尚未补齐的研究输入

仍缺少经审核的原始场景映射。文件 hash 一致只能证明数据字节一致，不能证明 calibration/pilot/confirmation 的场景独立。没有生成假 scene map，也没有真实 GPU G0 或实验结论。

提供真实场景证据后，CPU 准备命令保持为：

```bash
bash /home/fengyi/p01-transfer/p01-assets/prepare_inputs.sh /实际位置/reviewed-scene-map.json
```

它生成外部输入清单并检查容量，默认每集 50/50/80，不触发 GPU。后续 GPU 入口和标准见仓库 RUN_GPU.md；本次没有启动、等待或排队 GPU 实验。
