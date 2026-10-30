# P01 Astra Ultra 独立严格评审

评审日期：2026-10-30。对象：`/home/fengyi/CodeHome/Proc/src/ProC-SAM3`，分支 `codex/p01-scale-calibration`，HEAD `ba663c4bfe65c047932aaf0271d6ca27503b4c50`，基点 `234c87e58fb08f02ba1cf163b2a73f5330d462e9`。本评审由受派的 Astra Ultra 审查者完成，没有创建子代理。

**结论：方法值得做有界验证；当前版本不能取得“全部 CPU 工程合同均严格满足”的无保留通过。需要修复两个已复现的 P2 问题。没有发现已证实的 P0/P1 算法或主流程阻断。真实启动和完整研究均未通过，原因必须与 CPU 工程问题分别看待。**

| 判定层 | 严格结论 | 含义 |
|---|---|---|
| A：方法方案是否值得进入有界实验 | **有条件通过** | 同视图、同提示池、同先验下只改变条件签名，能够检验尺度条件是否有增量。先修下述工程缺口、补真实场景证据并通过真实 G0；不批准现在跳过门禁运行。 |
| B：本轮 CPU 工程是否严格通过 | **暂不通过，Request changes** | 原有 60 项 CPU 验收和 15 项迁移测试的通过记录可信，核心计算实现可保留。但预测写盘未执行 20 GiB 硬保护，报告未绑定预测使用的精确 bank receipt。二者是当前实现缺口，不是拿 GPU PENDING 否定 CPU 工作。修复后针对性复审即可，无理由重做方法架构。 |
| C：现在能否一键开始真实 P01 | **不能** | 有分阶段 launcher，不等于资产、真实 scene map、目标依赖、GPU 可运行性均就绪。当前配置默认关闭；本轮也没有 GPU 授权。 |
| D：完整 P01 研究是否通过 | **未通过，PENDING／证据不足** | 没有真实 G0、实际 504 特征、pilot、confirmation、效应 CI 或本轮在线资源测量。不能判假设成立，也不能判假设失败。 |
| E：耗时估计证据等级 | **历史耗时有记录；新硬件数字仅条件情景；小时级预算是经验预留** | 表中算术正确，措辞基本诚实。不能把 4090 的 5–8／8–14 分钟当本轮推理实测，更不能把 2–8 小时当完整实验的承诺上限。 |

这里的 B 是针对用户要求的严格合同签字，不是声称旧测试造假或全部工程不可用。两个 P2 均可局部修复；当前没有证据要求推翻 ECD-ProC 设计。

## 1. 范围、版本与证据纪律

遵循本轮“只读审查、不修复、不访问 GPU、不改外部工程”的授权。仓库源代码、配置、测试、分支均未改。新脚本、合成样本、报告只写当前审计目录。没有 SSH、远端写入、权重下载、MLLM/API 推理、GPU 排队、`nvidia-smi`、模型构造或模型推理，也没有重新扫描 30 GB 数据或重新哈希大权重。

所有新增涉及 torch 的命令均显式设置 `CUDA_VISIBLE_DEVICES=''`、`P01_ENABLE_GPU=0`、`PYTHONDONTWRITEBYTECODE=1`。类导入检查和合成检查均显示 `torch.cuda.is_initialized() == False`。导入原作者类时，依赖的 CUDA autocast 装饰器发出“CUDA 不可用、禁用 autocast”的警告；未构造模型、未建立 CUDA 上下文。该检查只证明本地现有环境可导入适配类。

父代理独立保存的 [provenance.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-review-v1/provenance.json) 核验了 40/40 项历史回执、小日志和测试源码哈希；它证明旧证据未变，不替代本次代码语义审查。历史生产代码摘要 `ca1816ca8606689c2f4077851d89819758b38dead540c6be86807563659eb9a3`，requirements 摘要 `027da4108d4aceaf6b4e25ef95499cbe2abcf274b5d7e6ee4d837b5a3c1fd5cb` 与交付记录一致。HEAD 与任务指定值一致，开始及审查后读取的工作区状态均干净。

读取了 graphify 技能说明；仓库无现成 `graphify-out/graph.json`。本轮明确禁止改仓库或扩张扫描，因此未安装 graphify、未生成图谱，直接沿入口与调用关系审查。

## 2. 已证实发现

### F1 — P2：CPU 预测写盘可以越过 20 GiB 保留线

**位置：** [calibration/run.py:311](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:311)，[launch.py:173](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/launch.py:173)，对照 [cache.py:90](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/cache.py:90) 与 [prepare.py:41](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/prepare.py:41)。

首次 G0 前，launcher 对全部缓存和预测做总量预算，这是正确的；恢复已有 G0 时，容量检查退为 `ensure_capacity(root, 0)`。后续 GPU 阶段对缺失缓存继续做容量检查，但 `decode_stage` 为全部 11 个方法执行 `np.save` 时没有任何容量保护。单独调用文档提供的 decode 命令也绕过 launcher 的预算检查。因此，在已有缓存后磁盘被其他文件占用、恢复时仅剩略高于保留量，或直接执行 CPU decode 时，预测会吃掉应保留的空间，直到操作系统报满。

**CPU 复现：** 在合法的两数据集、两视图合成 cache/bank 上，把 `shutil.disk_usage` 的返回值模拟为 `21,474,836,479` 字节，即 20 GiB−1。`decode_stage` 没有查询磁盘（调用次数 0），仍写出 44 个 NPY 文件、共 39,424 字节，并生成完成回执。见 [cpu-probe-results.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-review-v1/cpu-probe-results.json) 与 [reproduce_cpu.py](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-review-v1/reproduce_cpu.py)。这是模拟资源边界的确定性复现，**没有真的填满磁盘**。

**影响：** 违反明确的“保留至少 20 GiB，空间不足失败”合同；标准正常首次运行在空闲容量充足时不必然出错，但中断恢复及独立 CPU 入口缺少同等保护。原 `CODE_REVIEW.md` 所说“每次实际写入均保护 20 GiB”只适用于 cache 写入，不能覆盖预测文件。

**最小修复：** decode 开始前计算尚需写入的预测体积，检查输出所在文件系统；每批/每图写入前对本批剩余 NPY 体积加保留量检查，包含头部余量。保持已完成 GPU cache 可复用，失败结果保留。不要通过降低精度或删其他文件解决。

**验收：** 空闲量低于 `reserve + 本批写入量` 时，独立 decode 和 launcher 恢复路径都在写预测前失败；足额时成功；后续批次空间减少也会停止。确认未重跑已经核验通过的 GPU 阶段。该验收只需 CPU 小样本和模拟磁盘数值。

### F2 — P2：报告接受与预测来源不同的 bank receipt

**位置：** [report.py:15](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:15)、[report.py:47](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:47)、[report.py:83](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/report.py:83)，对照 [calibration/run.py:274](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:274)。

decode 身份保存了精确 `bank_receipt_sha256`，确认集 decode 也核验封存 bank。可是 `build_report` 仅检查预测与传入 bank 的共同实验身份，然后从传入 bank 读取支持场景数、回退行和校准成本，最终把该 bank 的 hash 写进报告；它未要求此 hash 等于生成预测时记录的 hash。

**CPU 复现：** 同一合成 calibration cache 分别拟合 `banks-a` 与 `banks-b`，二者合法且共同实验字段相同，但各自 run manifest/complete receipt 不同。用 A 解码，再向 report 传 B，报告仍成功，且继续正确标为 `PENDING_SYNTHETIC_ONLY`。A/B 的两个 receipt hash 与结果均已记录。

本例的两份 bank 来自**同一合成数据的确定性重复拟合**，没有证明数值预测或 mIoU 已经错误。它证明报告缺少精确来源绑定。如果后来误选另一次同配置校准运行，报告可以将预测 A 与 bank B 的支持/回退/成本证据拼在一起。标准 launcher 的默认固定路径降低了误用概率，文档中开放的独立 report 入口仍可触发。

**最小修复：** 生成报告前核对 `sha256(bank_run/complete.json) == prediction_manifest.identity.bank_receipt_sha256`；同时验证 bank 的 stage/kind 是 fit/calibration_banks。若要允许等价重建，应另做显式内容等价验证并记录，而不是默默接受任意同配置 bank。

**验收：** A 解码+A 报告成功；A 解码+B 报告必须失败，即使配置、输入清单、代码摘要相同；错误在写报告前出现。确认集进一步要求与封存 bank 绑定一致。此检查不需 GPU。

### F3 — P3：冻结配置校验未覆盖数据集集合

**位置：** [calibration/run.py:26](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/proc_research/calibration/run.py:26)，[config.json:4](/home/fengyi/CodeHome/Proc/src/ProC-SAM3/research/p01/config.json:4)。

把配置的 `datasets` 从 `['LoveDA','iSAID']` 改成 `['LoveDA']`，其余不变，`load_config` 仍通过；新增 CPU probe 已确认。标准 launcher 总是复制受版本控制的配置，所以默认入口不受影响；独立命令使用者却会得到“冻结政策已校验”的错误安全感，可能只运行并汇报一个数据集。

**影响与定级：** 这是低优先级的配置防误用缺口，不能据此声称当前默认配置漏跑了 iSAID，也不将它单独作为 B 否决原因。

**最小修复及验收：** 精确验证所需两数据集、无重复及预期顺序/集合；缺一个、重复或第三个数据集均在创建 run 前拒绝。若允许单数据集调试，明确标成非完整 P01，研究状态不得升级为两集确认通过。

没有发现 P0。没有足够证据认定 P1。以上发现均未在本轮修复。

## 3. 方法与核心实现审查

**方法有合理的可证伪命题，但创新空间有限。** 关键不是移植 ECD 本身，而是固定同一候选、同一两视图和同一全局先验后，真实尺度条件能否胜过全局 ECD、每尺度 RMS、朴素平均和每个打乱种子。当前冻结合同已经把这些区分写清楚。正结果只能先支持所选冻结样本，不能由两个数据集直接推导跨地域泛化或通用模型提升。

| 合同 | 当前实现判断与边界 |
|---|---|
| 冻结 SAM3、发布提示池，禁自动生成 | `model_options` 对发布池哈希做硬校验，缺失先失败；替换 MMSeg 接线和资产路径时检查原源码唯一匹配。未发现 MLLM/API/池生成入口被接入。 |
| 1008/504 各完整 ProC、串行、无 compile | `set_view` 重建对应 resolution 的 processor 并同步给 prompt bank/engine；不是先缩成 504 再交回固定 1008。ViT pre-hook 检查真实输入。静态路径合理，真实 encoder/decoder 支持仍待 G0。 |
| logit 插值、sigmoid、PGRF、类聚合顺序 | 原 reducer 保持先把 raw semantic/instance logits 插值回原图，再 sigmoid/加权；PGRF 与 max 聚合沿原方法。最终标签没有 bilinear。 |
| FP32 可回放缓存与身份 | 保存 prompt/class maps、presence、active、映射、geometry、预测及输入/权重/池/代码/配置摘要；外层 complete/index 绑定内部 receipt。配置/代码变化需新 run。未发现普通 cache 内容损坏可绕过现有校验的路径。 |
| 单视图 ECD 原式 | 本地独立 translation 与固定官方源码结构一致；保留双签名、uncalibrated 与 null 背景次序。历史 oracle parity 与手算案例有对应测试，未重复运行。 |
| 多视图适配 | 连续完整距离判别分数等权融合；分别作两种签名选择和 null 竞争，再按一致性/未校准状态回退 raw 两视图均值。没有平均类编号，也没有拿 log score 直接套概率阈值。明确称 ECD-ProC。 |
| 公平对照 | 全局/条件共用同一个 multi bank 中的质量先验、全局 uncalibrated、背景质量与超参；仅签名行变。单尺度三项、两视图均值/RMS/global/conditioned/uniform、shuffle 0/1/2 均有独立输出。 |
| 独立场景支持 | 按 scene 集合计数，每类/条件不足 5 则替换整行 R_a/R_e；不是拿像素或同场景切片数代替独立样本。 |
| 打乱对照 | 在 canonical scene/image/view 记录上置换并保留桶大小；要求每个原尺度内部混入两种条件，排除 identity 和单纯尺度改名。拟合与应用 assignment 均保存，种子分报。 |
| GT 边界 | 选择/拟合/推理只读图像、无标签缓存及场景证据；GT 像素读取集中在独立 evaluator。运输/标签清单辅助程序处理文件元数据，不将标签加入推理 manifest。 |
| 评价与统计 | LoveDA 0/255 ignore、1..7→0..6；iSAID 255 ignore；累加整图 confusion，按场景配对 bootstrap 2000 次、seed20261030，CI 下界>0才标 SUPERIOR；类跌幅>1pp限制结论。 |
| G0/confirmation 门禁 | 未通过真实 G0 不扩展；确认需要 seal 和精确 bank，且保留每张确认图像的 claim，改变名字不能重用曝光样本。普通缺授权调用在 preflight/model 之前拒绝。 |

官方来源核对限于固定提交，不使用浮动 main 替代：[ECD 官方源码](https://raw.githubusercontent.com/VILA-Lab/ActiveSAM/25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe/activesam/ecd.py)、[签名估计官方源码](https://raw.githubusercontent.com/VILA-Lab/ActiveSAM/25b84d8591a595f8bd8bef9ff4fbe1079ade9bfe/activesam/signatures.py)。本次阅读核对了算法结构；“本地文件字节未改”依据本地冻结哈希及旧验收回执，不把网页解析结果当作新做的远端逐字节 hash 对比。

仍应保留以下研究风险，它们**不是本轮已证实的代码错误**：

- 两次签名一致仍可能共同自信地错；需要真实 pilot 错误复核，不能用合成一致性测试证明校准有效。
- 每类/尺度的 5 场景门槛只是政策，不是统计功效保证。`select_splits` 按场景分桶后顺序截取图像，切片多的场景可能占满小预算；即便来源合法，也要检查实际独立场景数和条件回退比例。不能为了凑够支持捏造 scene ID，不能看确认标签补开发样本。
- 当前没有经审核真实 scene map。代码只能检查非空证据和 ID/hash 隔离，不能自动鉴别来源陈述真假；这点在文档中已披露，不是 CPU 编码未完成的证据。
- 发布提示池的构建图像、划分与模型修订不明；任何正结果仍不能宣称提示池已证实无泄漏。
- 实际 504 特征、两视图 G0 parity、显存、非确定性及 fresh-install 兼容性仍需后续授权实测；现有真实 GPU 结果不能由合成 shape metadata 或旧项目指标替代。
- 原执行引擎保留原作者的 OOM 缩小 prompt batch 重试路径。launcher 没有后台自动重试，但 runbook 的“模型失败不会自动重试”不能泛化成原引擎绝无内部 OOM 重试；该行为属于未改原方法，实际发生时应记录有效 batch 和资源口径。此次没有触发或测量它。

## 4. 启动与迁移判断

`launch.sh` 不是自动全量实验入口：pilot 计划最多覆盖冻结清单的 G0→calibration→fit→pilot→decode→evaluate→report；confirmation 是另一条显式命令，需 `--pilot-reviewed`。发布配置 `enabled=false`；运行需两个显式 GPU 开关。新增无开关 g0 调用返回 exit 1，失败发生在 `authorize`，没有进入 preflight 或模型。缺 scene map 的 `prepare_inputs.sh` 返回 exit 2，并明确表示未开始 split/experiment。

软件具备串联入口，但现在还不能承诺“一键开始真实 P01”：目标依赖未 clean-room 安装；权重/小元数据包尚未实际上传应用；真实 scene map 和隔离 splits 未生成；GPU G0 未测。修复 F1/F2 也不会自动补齐这些外部前提。

迁移脚本采用预览与显式 apply，预检查现有路径冲突、拒绝覆盖；`LoveDA`/`loveDA` 可通过别名兼容。读取的 compare/package helper 对路径、成员内容和哈希做检查，既有 15 项合成测试覆盖归档损坏、脚本冲突、大小写路径等情形。没有发现需要重新读取 26,626 个远端文件或重打包大权重的新证据。本评审接受原有“字节相同”的已记录范围，**不升级为 scene 独立或环境可用证明**。`scp` 和 tar 的目标冲突仍要按迁移说明检查；此次未执行。

报告目前主要汇入 CPU decode/calibration 资源，真实 GPU 数据在上游 GPU run 的 `resources.json` 和 `online_benchmark_*.json` 中。最终科研交付时必须把这些源文件或明确的路径/hash 一起交付，不能只给 `comparisons.json` 就声称在线显存/时延/编码数已完整分报。此处列为交付核对项，没有在缺少真实结果时伪造测量。

## 5. 耗时估计审计

读取了两个历史 `results/full/proc/{LoveDA,iSAID}/run.json`，确认历史总耗时和样本数：LoveDA `5839.825152397156 / 1669 = 3.4989964963 s/图`；iSAID `34382.89493918419 / 11644 = 2.9528422311 s/图`。这是旧源码版本、旧运行器的整段耗时，包含初始化/文件处理，不能作为本次 P01 G0 或新 decoder/cache 流程吞吐。

NVIDIA 官方资料支持 TITAN RTX 4608 CUDA cores、1770 MHz boost、672 GB/s、24 GB，以及 RTX 4090 82.6 FP32 TFLOPS、1008 GB/s、24 GB。按 `4608×1.77×2/1000` 得到 16.31232 TFLOPS，峰值 FP32 比例为 5.0636574074；带宽比例只有 1.5。这些是规格，不是该程序加速比。[TITAN RTX 官方规格](https://www.nvidia.com/en-us/titan/titan-rtx/)，[NVIDIA Ada 白皮书 Appendix A](https://images.nvidia.com/aem-dam/Solutions/geforce/ada/nvidia-ada-gpu-architecture.pdf)。

现行代码 encoder BF16、decoder FP32，关闭 TF32。旧 TITAN 的 BF16 路径、4090 的原生算术、内存/原图后处理和文件 IO 使单一 FP32 比值缺乏实际性能预测依据；5.06 倍也不是可靠加速上下界。文档已作这些限制，未发现将其冒称实测。

G0 的 28 个配对视图等价量算对了：4 图×3 组 direct/export/repeat 加 2 数据集×(5 warmup+3 timed)。在两个数据集图数相等、`t504/t1008∈[0.25,1]` 的**假设情景**下，重新算得：

| 图像总数 | TITAN 旧耗时外推，含 G0 | 再除峰值 FP32 比例得到的 4090 情景 |
|---|---:|---:|
| 360 | 26.076–41.722 min | 5.150–8.239 min |
| 600 | 42.206–67.529 min | 8.335–13.336 min |

与文档“约 26–42／42–68／5–8／8–14 分钟”一致；这是算术正确，**不是假设已验证**。0.25–1 不是已测 504 比例，更不是置信区间。两视图序列只需共享一次模型缓存，11 种方法不需要 11 次 SAM 推理，这一点正确。

证据等级可按以下方式使用：历史原机耗时＝可追溯历史测量；上表＝可复算但未验证的条件外推；4090 2–8 h、TITAN 3–9 h，以及 360 图的小时级范围＝经验排期缓冲，尚无本轮 CPU/存储分项测量支撑，不能认定为统计预测区间。它们不含环境准备、修复、scene 取证或人工 pilot 审核，因此也不是从当前状态到研究完成的总工期。待授权 G0 与少量 cache/CPU fit/decode 测量后应重估。

## 6. 实际执行与复现记录

本轮没有重复执行历史 60 项或 15 项完整测试。针对新疑点执行的命令和结果如下；Python 命令使用只读 `/home/fengyi/CodeHome/ProC-SAM3-repro/conda-env/bin/python`，经仓库 wrapper/ELF loader 运行，runtime 指向本审计目录。

| 命令/检查 | 退出码/结果 | 证据意义 |
|---|---|---|
| `git status --short`、`git rev-parse HEAD`、`git branch --show-current`、基点 diff stat | 0；工作区干净，HEAD/分支匹配 | 版本和只读审查基线 |
| 首次 `cat` 指向 `research/p01/decision.md`、`research/p01/METHOD.md` | 1，文件不存在；随后改读仓库根下两个文件，成功 | 单纯路径猜测错误，不是项目文件缺失或运行故障 |
| 主要文件的 `cat` / `nl` / `rg` 读取 | 0 | 直接审阅当前代码；没有仅复述旧 CODE_REVIEW |
| CPU 导入 `load_author_class()`，不实例化 | 0；`CachedSAM3OpenSegmentor`，CUDA initialized false | 现有环境类导入可行；不证明 GPU 模型可构造或推理 |
| `bash scripts/p01/python.sh …/reproduce_cpu.py` | 0；3 个 probe 全部完成 | F1/F2/F3 的独立合成复现；日志在 `cpu-probes.log` |
| `bash scripts/asset_transfer/prepare_inputs.sh /tmp/p01-astra-ultra-review-deliberately-missing-scene-map.json` | 2，预期拒绝 | 缺少 scene map 不启动准备/实验 |
| 禁 GPU 环境下 `bash scripts/p01/launch.sh g0 --settings research/p01/transfer.example.json` | 1，预期 PermissionError | 双开关门禁在 preflight/model 前拒绝 |
| Python 标准库读取两个旧 run.json、复算耗时 | 0 | 验证算术，未执行新推理 |
| 官方固定提交 ECD/signatures、NVIDIA 规格网页/PDF读取 | 工具成功 | 核对固定算法结构与硬件规格，不是 GPU 测量 |

直接的边界命令返回值及算术输出另存 [boundary-checks.json](/home/fengyi/CodeHome/Proc/runtime/p01/audit/astra-ultra-review-v1/boundary-checks.json)。新合成 fixture 的 encoder shape 都明确标为人工元数据，不能冒充实际 504/1008 encoder 证据。合成 GT 只供 evaluator，不读取真实数据标签。

主要已读文件：

- 需求：`docs/superpowers/plans/2026-10-30-p01-response-calibration.md`、`research/proposals/PROTOCOL.md`、`research/2026-10-30-4090-feasibility-review/REVIEW.md`（均位于 `/home/fengyi/CodeHome/Proc`）。按后续用户授权解释早期“只写计划”，没有误用旧文档否定已授权 CPU 实现。
- 当前仓库：`PROGRESS.md`、`BLOCKED.md`、`decision.md`、`METHOD.md`、`RUN_GPU.md`、`TRANSFER_4090.md`；`research/p01/config.json`、`run_manifest.json`、`ecd_parity.md`、`CODE_REVIEW.md`、冻结哈希与迁移/耗时记录。
- 核心全读：`proc_research/{baseline,cache,geometry,protocol,prepare,launch,evaluate,report,statistics,audit,paths}.py`，`calibration/{signatures,decode,run}.py`，`vendor/{ecd,signatures}.py` 及来源说明。
- 原方法关键调用链：`sam3_segmentor_cached.py` 的构造、PGRF、类聚合、predict；`core/result_reducer.py`；`core/grounding_execution_engine.py` 的单图编码、prompt 循环、输出整理与精度接口；`sam3/model/sam3_image_processor.py`；两数据集原配置。没有声称逐行评审整个 SAM3 模型。
- 脚本与测试：`scripts/p01/{python.sh,verify.py,launch.sh}`；`scripts/asset_transfer/{package.py,setup_links.py,compare_assets.py,select_label_metadata.py,prepare_inputs.sh}`；六个 `tests/research/test_*.py` 与三个 `tests/asset_transfer/test_*.py`；requirements。
- 外部只读小证据：两份旧运行 `run.json`。父代理核验的历史回执/日志/测试摘要详见 provenance.json。

未评审/未重复执行：完整 SAM3 底层架构与所有 CUDA kernel、任何真实模型前向/显存/吞吐、目标机驱动及 fresh-install、真实 scene map 真伪（尚缺）、真实预测精度、外部只读源工程的全套重跑、P02、已取消的完整大数据归档、可选二进制环境打包器的重新打包。理由分别是授权禁止、输入缺失、已收缩交付范围或与本次目标无关。未知不填 PASS，也不填已证实 FAIL。

## 7. 修复与后续验收次序

1. **先修 F1**：在独立 decode 和 launcher 恢复路径补齐预测写入容量保护，保存负例/恢复证据。
2. **再修 F2**：报告绑定精确 bank receipt，拒绝同配置但非预测来源的 bank；顺手增加 stage/kind 校验。
3. **补 F3**：冻结数据集集合校验。上述修复应形成新代码摘要；旧 CPU 通过记录保留，不能沿用旧摘要声称新代码已验收。
4. 完成针对性 CPU 回归和适当的工程验收后，才重新签 B。随后补 scene map/依赖/资产并取得新 GPU 授权，从小批真实 G0 开始；失败不扩展。
5. 通过 G0 后运行有界 pilot，检查条件支持、共同错误、每个打乱种子与资源分项，再按冻结协议决定单次 confirmation。均值未涨点不能自动终止校准检验；没有足够支持或 CI 证据就如实交付不确定/负结果。

当前没有修复实现，也没有运行任何真实 P01。允许继续的是局部 CPU 缺口修复与输入准备；不是把本报告当成真实实验已获通过或已获 GPU 授权。
