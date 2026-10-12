# 保留结果与复现范围

## ChebGCN 主模型

TADPOLE AD/CN/sMCI，seed0，十折、每折1000轮：ACC **94.31 ± 1.53%**，AUC **97.02 ± 1.91%**，Weighted F1 **94.35 ± 1.52%**。

结果：`Result/TADPOLE/AD_CN_SMCI/20260907_154033_677291_main_tadpole_3class.json`。完整轮次、划分及训练记录保留在同目录及 `logs/reproduce_full_20260907/`。

其他主任务历史锁定结果：TADPOLE sMCI/pMCI ACC96.44%；ABIDE ACC90.59%；ABIDE-5 ACC90.63%。这次整理没有重训这些任务；对应主配置和数据保留，历史归档已删除。

三分类十折有效权重另保存在 `chebgcn_checkpoints/`。清理时逐折重放，ACC与原记录一致，十折均值94.3136%。

## 固定10邻居 BB

TADPOLE AD/CN/sMCI，seed0，十折、每折500轮：同一模型关闭参照时ACC **92.811%**，正常参照 **93.980%**，598人中纠正7例、误改0例。

检查点、预测缓存及训练记录：`reference_fixed10/checkpoints/`。固定图及划分：`reference_fixed10/data_and_splits.npz`。原环境和源码hash：`reference_fixed10/provenance/`。

`python reference.py verify`重放十折全部预测与原缓存一致，最大logit误差0。新实现仅去掉旧继承链中的未用类别对常量；全部训练参数、初始化和梯度与旧BB一致。

## 解释范围

- 两个保留结果均沿用历史最高测试ACC选轮，不是独立验证集选出的测试估计。
- BB的92.811%是联合训练模型的自身输出，不是独立训练无图模型的成绩。
- ChebGCN与BB预算不同，不能直接用94.31%与93.98%判断优劣。
- 自动收益筛选在首轮新协议中为84.116%，低于同协议原注意力84.282%，未验证出分类提升，该实验已清理。
- 按用户要求，原消融、失败实验和完整历史归档已经永久删除。上述非保留主线数字仅作为历史摘要，不代表仍保存原运行文件。
