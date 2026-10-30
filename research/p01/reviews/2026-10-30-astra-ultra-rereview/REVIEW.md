# P01 Astra Ultra：F1/F2 修复独立复审

日期：2026-10-30。审查仓库 `/home/fengyi/CodeHome/Proc/src/ProC-SAM3`，分支 `codex/p01-scale-calibration`。精确审查范围：`e8e07cb7859fc4130d2ffddcb9c386bdd558721d..4013778dc6bd476bfa5182514498747e77c08ffe`。

**结论：F1、F2 两项 P2 均 CLOSED。本次收缩范围内的 CPU 工程严格复审 PASS，可以按用户授权推送该修复分支到指定 fork。没有新的阻断发现。此结论不授权合并、GPU 执行，也不表示真实 P01 或研究结论已通过。**

| 项目 | 本次判定 |
|---|---|
| F1：预测写盘缺少 20 GiB 保护 | **CLOSED**：全量预测预算、逐文件剩余预算及晚期元数据检查已接入实际 decode 路径；中途空间下降会失败并保留证据。 |
| F2：report 未绑定预测使用的精确 bank | **CLOSED**：fit stage/kind、精确 receipt hash、confirmation frozen hash 均在读取评价结果和创建报告前验证。 |
| CPU 工程验收 | **PASS**：修复代码、冻结回归、RED→GREEN 和完整 70 项 CPU 验收证据一致；新增独立边界验证通过。 |
| 推送修复分支 | **允许**：限用户已指定 fork/分支；评审没有执行推送或其他远端操作。 |
| F3：standalone datasets 配置校验 | **保留已知非阻断 P3，未修复**；本轮用户只要求修复两项 P2，不以此扩大否决范围。 |
| 方法方案 | 保持此前有条件可进入有界实验的判断，方法实现未因本修复改变。 |
| 当前真实启动／完整研究 | **NOT_READY／PENDING**：真实场景来源、目标依赖安装、真实 G0、504 特征和 pilot/confirmation 仍待后续授权及实测。 |

## 1. 版本、范围和证据核验

开始核查确认 HEAD 为 `4013778dc6bd476bfa5182514498747e77c08ffe`，分支正确、工作区干净。diff 中仅两个生产文件改变：`proc_research/calibration/run.py` 和 `proc_research/report.py`；新增 `tests/research/test_p2_regressions.py`，其余是整改计划、证据和 ledger。没有修改原方法、冻结配置、原有测试或迁移脚本。

重算当前 364 个生产源/配置/小资产文件的代码身份，结果为：

```text
code_sha256:
df52a7636ed6fb0de5a613ceea513c68632fdcca7f61c8d415ce900d45caab59
full_verification_sha256:
0e0b19a7b107e0e969d8145690e76d490dc857ca3a4af030cd66056239ae7098
new_regression_test_sha256:
33a36d9a6365d83114d81560ab6168cd57e1b200554af420749b45dabb5884f4
METHOD_sha256:
3d1a8ef6a4b98917c8dfc9fe778179093958e773544457ab12514659396a44b0
config_sha256:
73f79700a36f20d6741a0550e2aeb637973ba9b50b71ac857c8f15a41743ce23
```

核验 **31 项**报告/测试/回执/日志文件的摘要，全部一致，包括此前原报告、测试冻结记录、RED/F1 GREEN/完整 GREEN 日志、`cpu-p2-v1` 回执及其每条命令日志、测试源码、METHOD/config。此前报告仍为 `bbf4370c1e900595f0f7a0b3e44126ccaf0417369278968feb0f1f4aa688c513`，未被本轮改写。

读取实际 RED 日志：10 个测试有 9 条预期 AssertionError 失败记录，包括 stage/kind 两个 subcase；失败是应抛异常却未抛，不是 import/setup 错误。GREEN 使用完全相同 SHA256 的测试文件，10 个测试全部通过。完整验收日志显示 70 个测试、无 skips，全部验收命令 exit 0；三项损坏输入各 exit 1、恢复各 exit 0，六项 CLI help、冻结合同、原模型 diff 和 whitespace 检查均通过。没有只凭 ledger 的“通过”二字签字。

上述完整套件已经运行且源码/日志均未变化，本次没有无理由再跑 70 项或未变化的 15 项迁移测试。独立核验结果在 [evidence-verification.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/evidence-verification.json)。没有读取数据集内容或大权重来计算这些摘要。

## 2. F1：预算与实际写入路径

关键位置：[run.py:268](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:268)、[run.py:298](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:298)、[run.py:335](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:335)、[run.py:347](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:347)。

本次修复实质关闭了原来的漏检点：

- `prediction_budget` 在 `_cache_entries` 已核验 complete receipt 后读取各缓存的原图尺寸；按 `(dataset,image_id)` 计图，既不会把两视图算成两份最终预测，也不会把两个数据集中同名图像合并。尺寸必须为两个正整数，同图两视图尺寸不同则拒绝。
- 每图按全部 **11 种方法 × 原图 H×W × int64 8 字节**计入预测载荷；额外包含每个文件 64 KiB 头部/分配余量，以及总计 1 MiB+每图64 KiB 的元数据余量。预算没有只算单图、单数据集或某种方法。
- 全量检查位于 `create_run` 之前；独立 decode CLI 与 launcher 都调用此同一实现，不依赖首次 G0 的历史空闲量。
- 每个 NPY 写入前再次在目标路径所在文件系统检查剩余预算；写成功后按实际 array.nbytes 加文件余量扣减，当前文件又有独立最小预算保护。没有在写前错误扣掉尚未写入的预测。
- shuffle 元数据和最终 metadata/完成阶段前还有检查；异常不会生成 PASS 完成回执。已有 `preserve_incomplete_cpu` 恢复机制未改，失败目录保留，重新 decode 可以复用已经完成的推理 cache。

新增测试的不足容量、全图总量、NPY header、正常输出字节一致、中途失败和实际 launcher recovery helper 覆盖了原问题。它们依据输出文件和异常行为断言，不是仅断言“某个检查函数被调用”。原未修代码的真实 RED 日志说明它们确实能捕捉退化。

我又补了三个独立 CPU 边界：

| 新验证 | 实际结果 |
|---|---|
| 同一图只写完第一个方法后，将模拟空闲量降到 20 GiB−1 | 写出 **1** 个 NPY 后停止，下一方法未写；保留 manifest/部分输出，无 complete receipt。证明检查不只是每图一次。 |
| 全部44个 NPY写完后、最终元数据前，将模拟空闲量降到20 GiB−1 | **44** 个预测保留，`prediction_index.json` 与 complete receipt 均未生成。证明晚期容量下降不被当作已完成。 |
| 两数据集使用同名 image_id、四种非方形尺寸 `(8,12),(8,16),(12,12),(12,16)`，每图两视图 | 44个 NPY 完整生成；原始预测载荷49,280字节，全预算4,243,584字节、元数据余量1,310,720字节；正常完成回执存在。 |

这些是模拟磁盘返回值的小样本检查，**没有实际填满文件系统**。修复是保守预检查，并非原子磁盘预留：外部进程仍可能在检查与写入之间占用空间，底层 IO 失败可能留下未完成文件。README 已明确这一边界；本轮没有把普通 `disk_usage` 误说成能保证并发隔离，也不要求为关闭原 P2 引入文件系统配额/预分配机制。

**F1 判定：CLOSED。**

## 3. F2：精确 bank 与 confirmation 来源

关键位置：[report.py:15](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:15)、[report.py:17](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:17)、[report.py:19](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:19)、[report.py:22](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:22)、[report.py:89](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:89)。

现在报告同时核验三层约束：

1. bank 的 manifest stage 必须为 `fit`，完成回执 kind 必须为 `calibration_banks`；不能只靠 hash 对上就接受其他阶段。
2. bank `complete.json` 的实际 SHA256 必须等于预测身份中的 `bank_receipt_sha256`；缺少该字段也会拒绝。同配置、同数据但不同 run receipt 不再被自动视为预测来源。
3. 以预测的 `evaluated_split` 传入现有 `validate_bank_binding`，确认集报告还必须与 `frozen_bank_receipt_sha256` 一致；不会因为预测 run 自身 stage 是 `decode` 而漏掉 confirmation 检查。

上述检查全部早于读取评价目录 `provenance.json`，更早于创建报告输出。输出 provenance 使用已经验证过的 `bank_hash`，不是事后随意记下传入 bank。

已有冻结测试覆盖 A/A 正常、A/B 拒绝、缺失 binding、stage/kind 错误但 hash 对上，以及 confirmation 的正确/错误 seal 绑定。A/B仍是同一合成数据的重复拟合；这些测试证明来源约束，不声称它们本来产生了不同 mIoU。

新增独立验证特意传入**不存在的评价目录**：错误 bank、缺失 confirmation frozen hash、错误 frozen hash 三种情况均首先抛出预期的 ValueError，没有落到读取评价目录的 FileNotFoundError，也没有创建报告目录。这进一步验证了拒绝时机，不只验证最终“未生成报告”。

**F2 判定：CLOSED。**

## 4. 回归影响与已知范围

精确 diff 未改预测数学、签名估计、RMS/双签名/null/回退、scene/hash 隔离、shuffle、统计计算、真实模型适配、G0、freeze、confirmation claims 或原始 SAM3。正常预测路径只插入写入预算检查；report 只加强来源验证并复用已验证 hash。

正常足额路径的 44 个输出 NPY 与测试基准逐字节相同，正确 bank 的报告仍输出 `PENDING_SYNTHETIC_ONLY`。这些检查结合未变化的计算代码和完整 CPU 回归，支持“未引入已观察到的算法输出/冻结流程回归”；它们不证明真实 GPU 数值或准确率。

**没有新增 P0/P1/P2 发现。** 原 F3 仍是未修的非阻断 P3：[run.py:26](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:26) 的独立配置校验没有锁定 datasets 集合，标准 launcher 复制的两数据集配置仍正确。此前 F3 的影响和最小修复建议继续有效；本报告没有把它标成已关闭。

当前通过仅覆盖用户要求的 CPU 工程修复与复审；真实来源 scene map、目标机依赖、实际 504 编解码、G0 parity、显存/在线延迟、pilot/confirmation、研究效应和提示池来源限制均维持原状态。修复 receipt 绑定和容量检查不能消除这些外部未验证项，未验证 GPU 也不是本次 CPU 修复失败的理由。

## 5. 本次实际命令与产物

本次所有新文件只写 `/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/`。未编辑仓库代码、配置、测试、Git index/分支；未创建子代理；未使用 SSH 或远端写入；未构造 SAM3、未探测或初始化 GPU。

| 命令/工作 | 退出码和结果 |
|---|---|
| `git status --short`、`git rev-parse HEAD`、`git branch --show-current`、精确 diff/stat，读取相关 ledger、原评审和修复代码/测试/日志 | 0；HEAD正确、工作区干净，改动范围吻合 |
| `python3 …/astra-ultra-rereview-v1/verify_evidence.py` | **0**；31项小证据摘要匹配，364个生产文件重算摘要一致；核对RED/GREEN及70项旧验收记录 |
| `bash scripts/p01/python.sh …/astra-ultra-rereview-v1/probe_boundaries.py` | **0**；新增6个独立边界probe全部PASS，`cuda_initialized=false` |

CPU probe 环境显式设置：`CUDA_VISIBLE_DEVICES='' P01_ENABLE_GPU=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`；Python 为只读 `/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python`；`P01_RUNTIME` 指向本审计目录的 runtime。没有调用模型构造器、CUDA 可用性检测或真实推理。合成 encoder shape 只是 fixture 元数据，不能当实际编码尺寸证据。

主要新产物：

- [commands.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/commands.json)：本次两个实际验证命令及退出码。
- [evidence-verification.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/evidence-verification.json)：逐文件核验和生产摘要。
- [boundary-results.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/boundary-results.json)：6个独立边界结果。
- [probe_boundaries.py](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-rereview-v1/probe_boundaries.py)：可复核的新验证源码；输出目录按唯一运行保留，重跑应使用新目录。
- `evidence-verification.log`、`boundary-probes.log` 与 `synthetic-boundaries/`：完整本地输出及保留的成功/失败合成证据。

本次是对已整改两个 P2 的独立复审，没有重复整个初次项目审查、历史60/15套验收、大数据哈希、GPU测试或耗时估计调研。没有发现需要扩大范围的新证据。

**签署：对 reviewed HEAD `4013778dc6bd476bfa5182514498747e77c08ffe`、生产摘要 `df52a763…ab59`，关闭 F1/F2，CPU 工程 PASS。可以归档此报告并推送用户授权的修复分支；若后续只增加本次评审/交付记录，生产摘要应保持一致。真实实验与科研结论继续 PENDING。**
