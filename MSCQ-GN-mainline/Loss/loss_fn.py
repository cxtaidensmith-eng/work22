import torch
from torch import nn
import torch.nn.functional as F


def orthogonality_loss(features, epsilon=1e-8):
    """Encourage different class-semantic embeddings to be distinct."""
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


class MSCQGNLoss(object):
    def __init__(self, DATASET_Dict, Device, rate=0.001, label_smoothing=0.05,
                 aux_loss_weight=1.0, aux_supervision='ovr'):
        super().__init__()
        self.DATASET_Dict = DATASET_Dict
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']

        self.CE_loss = nn.CrossEntropyLoss(weight=self.weight, label_smoothing=label_smoothing).to(Device)
        self.rate = rate
        self.aux_loss_weight = float(aux_loss_weight)
        if aux_supervision not in {'ovr', 'multiclass'}:
            raise ValueError(f'Unknown auxiliary supervision: {aux_supervision}')
        self.aux_supervision = aux_supervision

        self.aux_loss_dict = nn.ModuleDict()
        for i in range(self.Label_num):
            weight_matrix = torch.empty(2).to(Device)
            weight_matrix[1] = self.weight[i]
            weight_matrix[0] = torch.sum(self.weight) - weight_matrix[1]

            self.aux_loss_dict[f'aux_loss_{i}'] = nn.CrossEntropyLoss(weight=weight_matrix, label_smoothing=label_smoothing).to(Device)

    def __call__(self, output, Y, mask, _Label_embedding, _Auxi_classifier_output):
        ce_loss = self.CE_loss(output[mask], Y[mask])
        if not _Auxi_classifier_output:
            if self.aux_loss_weight != 0:
                raise ValueError('Readout without auxiliary heads requires auxiliary weight zero')
            if not _Label_embedding:
                if self.rate != 0:
                    raise ValueError('Single-query pooling requires orthogonality weight zero')
                return ce_loss
            return ce_loss + self.rate * orthogonality_loss(_Label_embedding)

        if self.aux_supervision == 'multiclass':
            if len(_Auxi_classifier_output) != self.Label_num:
                raise ValueError('Expected one generic classifier per query')
            aux_loss = sum(self.CE_loss(head[mask], Y[mask]) for head in _Auxi_classifier_output)
            return ce_loss + self.aux_loss_weight * aux_loss + self.rate * orthogonality_loss(_Label_embedding)

        Y_one_hot = F.one_hot(Y, num_classes=self.Label_num)
        Y_one_hot = Y_one_hot.transpose(0, 1).reshape(self.Label_num, -1)

        aux_loss = 0
        for i in range(self.Label_num):
            aux_loss_ = self.aux_loss_dict[f'aux_loss_{i}'](_Auxi_classifier_output[i][mask], Y_one_hot[i][mask])
            aux_loss = aux_loss + aux_loss_

        orth_loss = orthogonality_loss(_Label_embedding)

        return ce_loss + self.aux_loss_weight * aux_loss + self.rate * orth_loss
