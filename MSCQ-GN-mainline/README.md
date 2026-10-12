# 框架 C：内容模态条件协作传播

本分支保存截至 2026-10-12 的框架 C 实现、四任务历史实验指标，以及最新六组控制实验的代码、测试和汇总。代码目录是 `MSCQ-GN-mainline/`；仓库根目录已有代码属于原远程版本。

当前主模型是 `Model/relation_sheaf.py` 的 `RelationSheafModel(mode='content')`，训练入口是 `relation_experiment.py`。`main.py` 是历史 ChebGCN 入口。后续实验变体没有替换 C。

## 网络流程

模态独立编码 → 各原始模态建立余弦 top-k 对称候选图 → 每个关系来源传播每个内容模态 → 内容模态条件的发送/接收映射与双扩散 → 关系来源等权平均及残差更新 → 患者内部模态融合与类别查询读取 → 与自身多模态分支相加 → MLP 分类。

候选拓扑固定，边上的标量权重随训练变化。TAD 三分类有六个关系来源和六个内容模态，共 36 条关系来源×内容路径。发送/接收映射按内容模态条件化，在患者和关系来源间共享。

## 当前 C 的构图与传播顺序

以 TAD 三分类的六模态任务为例：

1. 训练准备阶段，用各模态的处理后原始特征分别构建六张患者候选关系图。各图按余弦相似度选 top-10 邻居，再与反向边取并集。对应 `relation_experiment.py:65` 与 `Model/relation_sheaf.py:16–38`。
2. 模态编码得到六组内容表示；此时尚未进行患者内部的跨模态融合。
3. 六张候选关系图与六组内容表示进入同一个协作传播模块。每张关系图都传播全部内容模态，共六类关系×六种内容的 36 条路径。
4. 每种内容对全部关系来源的更新等权平均，并残差更新对应模态表示。
5. 更新后的模态表示才进入模态 Transformer 融合、类别查询读取，与患者自身语义分支相加，经 MLP 分类。对应 `Model/relation_sheaf.py:225–239`。

候选邻居和支持边固定；每次前向会根据当前内容表示计算这些边上的权重。当前 C 没有融合后重新选择邻居或重建六张图，也没有 `F1→G1→P1→F2→G2→P2` 的交替循环。传播模块调用一次，内部含两次协作扩散；两次扩散之间没有融合或重建图。

模态门控是编码前后使用的可学习模态参数；残差传播强度是另一个可学习参数。二者不能描述成“邻居消息动态生成门控反馈”。候选图、内容相关边权和发送/接收映射是不同的对象。

仓库中保留的 `alternating_cooperative.py` 是历史实验变体，其 A1 顺序为 `P→F1→F2→P→F3→F4→F5`，同样使用固定候选图，只在第二次传播重算边权。它没有替换当前 C。

## 安装与检查

```powershell
cd MSCQ-GN-mainline
python -m pip install -r requirements.txt
python -B -m unittest discover -s tests -p test_relation_sheaf.py -v
python -B -m unittest discover -s tests -p test_relation_decay.py -v
```

上述检查使用合成张量，无需患者数据或模型权重。绘图脚本另需安装 `matplotlib`，部分历史绘图脚本需自行提供字体文件。

## 训练所需外部资料

本次新增目录不携带处理后患者表、预测、划分/候选图缓存或训练权重。训练前须自行准备相同版本的以下文件：

- `DATASET/TADPOLE/{任务}_ADNI_processed_standard_data.csv` 与 `{任务}_ADNI_modal_feat_dict.npy`；任务包含 `AD_CN_SMCI` 和 `SMCI_PMCI`。
- `DATASET/ABIDE/` 和 `DATASET/ABIDE-5/` 内的 `ABIDE_processed_data_modal.csv` 与 `ABIDE_modal_feat_dict.npy`。
- TAD 三分类还强制读取 `reference_fixed10/data_and_splits.npz`，核对原患者顺序、标签和十折掩码；它也是外部资料。

数据应放入本代码目录下的对应路径。不能仅凭仓库根目录旧数据的文件名认定其与历史实验版本一致。

资料齐备后，运行保留的 TAD 三分类 C 配置：

```powershell
python relation_experiment.py --config Config/main_tadpole_3class.ini --modes content --graph-weight-decay 0 --training-seed 0 --cuda 0 --output Result/c_tad3_new
```

CPU 使用 `--cuda -1`。该命令默认完整十折、每折 1000 轮，不会覆盖已有结果目录。四任务各自的保留配置涉及专用模型/损失/优化器工厂覆盖，不能只用导出 INI 宣称完整复现。对应源码包括 `tad_binary_joint_tune.py`、`tad_binary_loss_tune.py`、`abide_fine_tune.py`、`abide5_fine_tune.py` 和 `abide5_tie_completion.py`。

## 历史结果与口径

以下为 seed0、原十折协议的折均指标：

| 任务 | 保留配置 | 最佳 ACC | 最佳 macro-F1 |
| --- | --- | ---: | ---: |
| TADPOLE AD/CN/SMCI | C content，图参数衰减 0 | 94.980226% | 93.924643% |
| TADPOLE SMCI/PMCI | J_DN | 97.201258% | 89.198430% |
| ABIDE ADS/CN | R_L115 | 89.438349% | 89.356792% |
| ABIDE-5 ADS/CN | BASE | 91.786955% | 91.715098% |

实验沿用全队列图的传导式设置，按每折首次最高测试 ACC 选轮；测试折已用于调参与探索，没有独立最终测试集。这些是探索协议下的历史结果，不是未参与开发的最终泛化估计。TAD 三分类 C 的三个训练种子 ACC 均值为 94.146893%。

机器可读指标：[metrics.csv](published_results/c_framework_20260929/metrics.csv)、[fold_metrics.csv](published_results/c_framework_20260929/fold_metrics.csv)、[metrics.json](published_results/c_framework_20260929/metrics.json)。导出的历史 INI 位于同目录 `configs/`。

最新控制实验汇总位于 `published_results/`，包括读取器、共享映射、互选关系、自身分支优化、匹配共享映射和去除内容相似度；各目录保留折级/总体统计与原结果说明。六组 `summary.json` 的本机绝对路径已转换为相对路径；引用的私有预测和缓存文件不随本次提交上传。

这些历史控制实验的训练/分析入口还依赖原 `Result/` 内的实验计划、参考权重、预测、协议、审计和图缓存。仅准备上列患者 CSV、模态字典和历史划分文件不足以运行所有历史控制入口；`published_results/` 的汇总不能替代原实验资产。

## 当前可视化对应的模型

已核对当前边权分解、内容映射交换、关系来源邻居重叠和关系×内容路径干预脚本。它们对应同一个 `RelationSheafModel(mode='content')`；模型推断脚本读取 `Result/relation_mechanism_nodecay/seed_0/content/fold_XX/best.pt` 的原十折 C 检查点，邻居重叠分析使用同一实验的原始候选图。发布代码与当前本地代码一致，仅统一文本换行。

最新框架图：[10 月 12 日并列模态关系图 v4](diagrams/framework_c_20261012/framework_c_parallel_modality_zh_v4.png)，[对应图注与示意边界](diagrams/framework_c_20261012/figure_notes_zh_v4.txt)。10 月 11 日 v1–v3 也在 `diagrams/framework_c_20261011/` 中保留。图说明和生成提示中的本机参考路径已省略。

这次补充的是最新架构示意图；边权、映射交换和邻居重叠等实验输出图及其完整诊断资产没有随本次提交上传。它们的分析/绘图脚本已在代码目录中保留。

## 发布范围与追溯

本次提交在原远程 `paper-development` 基础上新增独立目录，保留远程原有历史。本次新增内容为当前源码、配置、合成数据测试、汇总指标和框架图；提交在独立副本完成，原工作区文件、分支和 Git 索引保持原状。

`PUBLISH_MANIFEST.json` 记录本次包内文件的 SHA-256 和本地来源提交。它是本次发布清单，不是原 2026-09-29 完整实验快照的 manifest。当前包不含完整预测/检查点证据，不能使用原 `snapshot_c_framework.py --verify` 宣称整个历史证据包验证通过，也未重新训练产生表中指标。
