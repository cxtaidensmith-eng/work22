import torch
from torch import nn
import torch.nn.functional as F


class criterion_loss(object):
    def __init__(self, DATASET_Dict, Device):
        super(criterion_loss, self).__init__()
        self.DATASET_Dict = DATASET_Dict
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']

        self.CE_loss = nn.CrossEntropyLoss(weight=self.weight).to(Device)

        self.aux_loss_dict = nn.ModuleDict()
        for i in range(self.Label_num):
            weight_matrix = torch.empty(2).to(Device)
            weight_matrix[1] = self.weight[i]
            weight_matrix[0] = torch.sum(self.weight) - weight_matrix[1]

            self.aux_loss_dict[f'aux_loss_{i}'] = nn.CrossEntropyLoss(weight=weight_matrix).to(Device)
    
    def __call__(self, output, Y, mask, _Label_embedding, _Auxi_classifier_output):
        ce_loss = self.CE_loss(output[mask], Y[mask])

        Y_one_hot = F.one_hot(Y, num_classes=self.Label_num)
        Y_one_hot = Y_one_hot.transpose(0, 1).reshape(self.Label_num, -1)

        aux_loss = 0
        for i in range(self.Label_num):
            aux_loss_ = self.aux_loss_dict[f'aux_loss_{i}'](_Auxi_classifier_output[i][mask], Y_one_hot[i][mask])
            aux_loss = aux_loss + aux_loss_
        
        return ce_loss + aux_loss


def orthogonality_loss(features, epsilon=1e-8):
    """
    计算给定特征列表中所有特征矩阵之间的正交性损失，去掉循环。
    :param features: 包含多个特征矩阵的列表，每个特征矩阵形状为 (batch_size, feature_dim)
    :return: 正交性损失
    """
    # 将所有特征矩阵堆叠成一个大的矩阵，形状为 (num_features, batch_size, feature_dim)
    stacked_features = torch.stack(features)  # (num_features, batch_size, feature_dim)

    # L2 归一化
    norm = torch.norm(stacked_features, dim=2, keepdim=True)
    norm = torch.clamp(norm, min=epsilon)  # 避免除以零
    normalized_features = stacked_features / norm

    # 计算特征矩阵之间的点积，形状为 (num_features, num_features, batch_size, batch_size)
    dot_products = torch.einsum('nbi,mbj->nmij', normalized_features, normalized_features)

    # 提取所有不同特征矩阵之间的点积
    num_features = len(features)
    indices = torch.triu_indices(num_features, num_features, 1)
    pairwise_dot_products = dot_products[indices[0], indices[1]]

    # 计算 Frobenius 范数并作为损失
    loss = torch.norm(pairwise_dot_products, p='fro') ** 2

    return loss


def orthogonality_lossv2(features, epsilon=1e-8):
    """
    计算给定特征列表中所有特征矩阵之间的正交性损失。
    :param features: 包含多个特征矩阵的列表，每个特征矩阵形状为 (batch_size, feature_dim)
    :return: 正交性损失
    """
    loss = 0
    num_features = len(features)

    for i in range(num_features):
        for j in range(i + 1, num_features):
            # 对两个特征进行 L2 归一化
            F1_norm = features[i] / torch.clamp(torch.norm(features[i], dim=1, keepdim=True), min=epsilon)
            F2_norm = features[j] / torch.clamp(torch.norm(features[j], dim=1, keepdim=True), min=epsilon)

            # 计算点积
            dot_product = torch.mm(F1_norm, F2_norm.t())

            # 累加 Frobenius 范数的平方
            loss += torch.norm(dot_product, p='fro') ** 2

    return loss


class criterion_lossv2(object):
    def __init__(self, DATASET_Dict, Device, rate=0.001):
        super(criterion_lossv2, self).__init__()
        self.DATASET_Dict = DATASET_Dict
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']

        self.CE_loss = nn.CrossEntropyLoss(weight=self.weight).to(Device)
        self.rate = rate

        self.aux_loss_dict = nn.ModuleDict()
        for i in range(self.Label_num):
            weight_matrix = torch.empty(2).to(Device)
            weight_matrix[1] = self.weight[i]
            weight_matrix[0] = torch.sum(self.weight) - weight_matrix[1]

            self.aux_loss_dict[f'aux_loss_{i}'] = nn.CrossEntropyLoss(weight=weight_matrix).to(Device)

    def __call__(self, output, Y, mask, _Label_embedding, _Auxi_classifier_output):
        ce_loss = self.CE_loss(output[mask], Y[mask])

        Y_one_hot = F.one_hot(Y, num_classes=self.Label_num)
        Y_one_hot = Y_one_hot.transpose(0, 1).reshape(self.Label_num, -1)

        aux_loss = 0
        for i in range(self.Label_num):
            aux_loss_ = self.aux_loss_dict[f'aux_loss_{i}'](_Auxi_classifier_output[i][mask], Y_one_hot[i][mask])
            aux_loss = aux_loss + aux_loss_

        orth_loss = orthogonality_lossv2(_Label_embedding)

        return ce_loss + aux_loss + self.rate * orth_loss