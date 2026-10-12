# 框架 C：内容模态条件协作传播

本分支保存截至 2026-10-12 的框架 C 实现、四任务历史实验指标，以及最新六组控制实验的代码、测试和汇总。代码目录是 `MSCQ-GN-mainline/`；仓库根目录已有代码属于原远程版本。

当前主模型是 `Model/relation_sheaf.py` 的 `RelationSheafModel(mode='content')`，训练入口是 `relation_experiment.py`。`main.py` 是历史 ChebGCN 入口。后续实验变体没有替换 C。

## 网络流程

模态独立编码 → 各原始模态建立余弦 top-k 对称候选图 → 每个关系来源传播每个内容模态 → 内容模态条件的发送/接收映射与双扩散 → 关系来源等权平均及残差更新 → 患者内部模态融合与类别查询读取 → 与自身多模态分支相加 → MLP 分类。

候选拓扑固定，边上的标量权重随训练变化。TAD 三分类有六个关系来源和六个内容模态，共 36 条关系来源×内容路径。发送/接收映射按内容模态条件化，在患者和关系来源间共享。

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

## 发布范围与追溯

本次提交在原远程 `paper-development` 基础上新增独立目录，保留远程原有历史。本次新增内容为当前源码、配置、合成数据测试、汇总指标和框架图；提交在独立副本完成，原工作区文件、分支和 Git 索引保持原状。

`PUBLISH_MANIFEST.json` 记录本次包内文件的 SHA-256 和本地来源提交。它是本次发布清单，不是原 2026-09-29 完整实验快照的 manifest。当前包不含完整预测/检查点证据，不能使用原 `snapshot_c_framework.py --verify` 宣称整个历史证据包验证通过，也未重新训练产生表中指标。
