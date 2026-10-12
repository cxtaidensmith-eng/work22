# work22：框架 C

本分支保存当前框架 C 的代码、配置、合成数据测试和实验汇总。

请从 [框架 C 说明与运行方法](MSCQ-GN-mainline/README.md) 开始。

**当前 C 先按原始模态特征建立六张候选患者图，再进行关系×内容协作传播，最后融合、读取与分类。** 候选支持边固定，边权根据当前内容计算；主模型没有反复“融合→构图→传播”的循环。完整步骤见 [当前 C 的执行顺序](MSCQ-GN-mainline/README.md#当前-c-的构图与传播顺序)。

- 核心模型：[RelationSheafModel](MSCQ-GN-mainline/Model/relation_sheaf.py)，使用 `mode='content'`。
- 训练入口：[relation_experiment.py](MSCQ-GN-mainline/relation_experiment.py)。
- 实验汇总：[published_results](MSCQ-GN-mainline/published_results/)。
- 框架图：[diagrams](MSCQ-GN-mainline/diagrams/)。

本次发布日期为 2026-10-12。完整训练需要另行准备处理后数据与历史划分，详见框架 C 说明。根目录原有代码属于此前远程版本。
