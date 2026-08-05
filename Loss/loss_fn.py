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
    def __init__(self, DATASET_Dict, Device, rate=0.001, label_smoothing=0.05):
        super(criterion_lossv2, self).__init__()
        self.DATASET_Dict = DATASET_Dict
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']

        self.CE_loss = nn.CrossEntropyLoss(weight=self.weight, label_smoothing=label_smoothing).to(Device)
        self.rate = rate

        self.aux_loss_dict = nn.ModuleDict()
        for i in range(self.Label_num):
            weight_matrix = torch.empty(2).to(Device)
            weight_matrix[1] = self.weight[i]
            weight_matrix[0] = torch.sum(self.weight) - weight_matrix[1]

            self.aux_loss_dict[f'aux_loss_{i}'] = nn.CrossEntropyLoss(weight=weight_matrix, label_smoothing=label_smoothing).to(Device)

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


class criterion_query_free_multibranch(object):
    """Main CE plus one full multi-class auxiliary CE per latent branch.

    V1 intentionally has no orthogonality term.  Each branch predicts the full
    class label, so branch indices are not assigned to individual diagnoses.
    """

    def __init__(self, DATASET_Dict, Device, label_smoothing=0.05, aux_weight=1.0):
        super(criterion_query_free_multibranch, self).__init__()
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']
        self.aux_weight = float(aux_weight)
        self.main_loss = nn.CrossEntropyLoss(
            weight=self.weight,
            label_smoothing=label_smoothing,
        ).to(Device)
        self.aux_losses = nn.ModuleList([
            nn.CrossEntropyLoss(
                weight=self.weight,
                label_smoothing=label_smoothing,
            ).to(Device)
            for _ in range(self.Label_num)
        ])

    def __call__(self, output, Y, mask, _Branch_embedding, _Auxi_classifier_output):
        if len(_Auxi_classifier_output) != self.Label_num:
            raise ValueError(
                f'Expected {self.Label_num} latent auxiliary outputs, '
                f'got {len(_Auxi_classifier_output)}'
            )
        main_loss = self.main_loss(output[mask], Y[mask])
        aux_loss = output.new_zeros(())
        for criterion, aux_output in zip(self.aux_losses, _Auxi_classifier_output):
            if aux_output.size(-1) != self.Label_num:
                raise ValueError(
                    f'Expected a {self.Label_num}-class latent auxiliary head, '
                    f'got shape {tuple(aux_output.shape)}'
                )
            aux_loss = aux_loss + criterion(aux_output[mask], Y[mask])
        return main_loss + self.aux_weight * aux_loss


class criterion_query_pool_no_orth(object):
    """Original one-vs-rest auxiliary supervision without orthogonality.

    This is the paired control for criterion_lossv2: main CE, class weights,
    label smoothing, and the three binary auxiliary losses are unchanged; only
    the orthogonality term is absent.
    """

    def __init__(self, DATASET_Dict, Device, label_smoothing=0.05):
        super(criterion_query_pool_no_orth, self).__init__()
        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']
        self.main_loss = nn.CrossEntropyLoss(
            weight=self.weight,
            label_smoothing=label_smoothing,
        ).to(Device)
        self.aux_losses = nn.ModuleList()
        for label_idx in range(self.Label_num):
            binary_weight = torch.empty(2, device=Device)
            binary_weight[1] = self.weight[label_idx]
            binary_weight[0] = torch.sum(self.weight) - binary_weight[1]
            self.aux_losses.append(
                nn.CrossEntropyLoss(
                    weight=binary_weight,
                    label_smoothing=label_smoothing,
                ).to(Device)
            )

    def __call__(self, output, Y, mask, _Label_embedding, _Auxi_classifier_output):
        if len(_Auxi_classifier_output) != self.Label_num:
            raise ValueError(
                f'Expected {self.Label_num} query-pool auxiliary outputs, '
                f'got {len(_Auxi_classifier_output)}'
            )
        main_loss = self.main_loss(output[mask], Y[mask])
        one_vs_rest_targets = F.one_hot(Y, num_classes=self.Label_num).transpose(0, 1)
        aux_loss = output.new_zeros(())
        for label_idx, (criterion, aux_output) in enumerate(
            zip(self.aux_losses, _Auxi_classifier_output)
        ):
            if aux_output.size(-1) != 2:
                raise ValueError(
                    f'Expected a binary query-pool auxiliary head, '
                    f'got shape {tuple(aux_output.shape)}'
                )
            aux_loss = aux_loss + criterion(
                aux_output[mask], one_vs_rest_targets[label_idx][mask]
            )
        return main_loss + aux_loss


class criterion_boundary_aware_multibranch(object):
    """Three fixed pairwise diagnostic-boundary auxiliary objectives.

    The main three-class objective and network forward are unchanged.  Each
    binary auxiliary head is supervised only on the two diagnoses defining its
    boundary; the excluded diagnosis still contributes through the main loss.
    """

    BOUNDARIES = (
        ("CN_SMCI", 1, 2),
        ("SMCI_AD", 2, 0),
        ("CN_AD", 1, 0),
    )

    def __init__(
        self,
        DATASET_Dict,
        Device,
        label_smoothing=0.05,
        aux_weight=1.0,
        active_boundaries=None,
    ):
        super(criterion_boundary_aware_multibranch, self).__init__()
        if DATASET_Dict['Class_Num'] != 3:
            raise ValueError('Boundary-aware v1 requires exactly three classes')
        if float(aux_weight) != 1.0:
            raise ValueError('Boundary-aware v1 fixes auxiliary loss weight at 1.0')

        class_names = DATASET_Dict.get('Class_Names')
        if class_names is not None and list(class_names) != ['AD', 'CN', 'SMCI']:
            raise ValueError(
                'Boundary-aware v1 requires class order [AD, CN, SMCI], '
                f'got {list(class_names)}'
            )

        self.weight = DATASET_Dict['Label_Weight']
        self.Label_num = DATASET_Dict['Class_Num']
        self.aux_weight = 1.0
        boundary_names = tuple(name for name, _, _ in self.BOUNDARIES)
        self.active_boundaries = (
            boundary_names
            if active_boundaries is None
            else tuple(active_boundaries)
        )
        unknown_boundaries = set(self.active_boundaries) - set(boundary_names)
        if unknown_boundaries:
            raise ValueError(
                f'Unknown active boundaries: {sorted(unknown_boundaries)}'
            )
        if len(self.active_boundaries) != len(set(self.active_boundaries)):
            raise ValueError('active_boundaries must not contain duplicates')
        self.main_loss = nn.CrossEntropyLoss(
            weight=self.weight,
            label_smoothing=label_smoothing,
        ).to(Device)
        self.boundary_losses = nn.ModuleDict()
        for name, lower_class, higher_class in self.BOUNDARIES:
            pair_weight = torch.stack((
                self.weight[lower_class],
                self.weight[higher_class],
            )).to(Device)
            self.boundary_losses[name] = nn.CrossEntropyLoss(
                weight=pair_weight,
                label_smoothing=label_smoothing,
            ).to(Device)

    @classmethod
    def boundary_masks_and_targets(cls, labels, mask):
        mask = mask.bool()
        result = {}
        for name, lower_class, higher_class in cls.BOUNDARIES:
            boundary_mask = mask & (
                (labels == lower_class) | (labels == higher_class)
            )
            target = (labels[boundary_mask] == higher_class).long()
            result[name] = {
                'mask': boundary_mask,
                'target': target,
                'lower_class': lower_class,
                'higher_class': higher_class,
            }
        return result

    def compute(
        self,
        output,
        Y,
        mask,
        _Label_embedding,
        _Auxi_classifier_output,
    ):
        if len(_Auxi_classifier_output) != len(self.BOUNDARIES):
            raise ValueError(
                f'Expected {len(self.BOUNDARIES)} boundary auxiliary outputs, '
                f'got {len(_Auxi_classifier_output)}'
            )

        main_loss = self.main_loss(output[mask], Y[mask])
        components = {'main': main_loss}
        boundary_data = self.boundary_masks_and_targets(Y, mask)
        auxiliary_sum = output.new_zeros(())
        for (name, _, _), auxiliary_output in zip(
            self.BOUNDARIES, _Auxi_classifier_output
        ):
            if auxiliary_output.size(-1) != 2:
                raise ValueError(
                    f'Expected a binary head for boundary {name}, '
                    f'got shape {tuple(auxiliary_output.shape)}'
                )
            boundary_mask = boundary_data[name]['mask']
            target = boundary_data[name]['target']
            if target.numel() == 0 or target.unique().numel() != 2:
                raise ValueError(
                    f'Boundary {name} must contain both classes in the loss mask'
                )
            boundary_loss = self.boundary_losses[name](
                auxiliary_output[boundary_mask], target
            )
            components[name] = boundary_loss
            if name in self.active_boundaries:
                auxiliary_sum = auxiliary_sum + boundary_loss

        components['auxiliary_sum'] = auxiliary_sum
        total = main_loss + self.aux_weight * auxiliary_sum
        components['total'] = total
        return total, components

    def __call__(
        self,
        output,
        Y,
        mask,
        _Label_embedding,
        _Auxi_classifier_output,
    ):
        total, _ = self.compute(
            output,
            Y,
            mask,
            _Label_embedding,
            _Auxi_classifier_output,
        )
        return total
