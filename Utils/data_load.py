import os

os.environ.setdefault('OMP_NUM_THREADS', '1')

import pandas as pd
import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from .graph_load import feature_to_adj


def load_dataset(Feature_Data_path, Feature_dict_path, Device, Class_names_list, Shuffle=True, Random_seed=0,
                 k_nearest_neighobrs=7, rm_common_neighbors=2, train_size=0.9):
    Feature_dict = np.load(Feature_dict_path, allow_pickle=True).item()
    Feature_Data = pd.read_csv(Feature_Data_path, low_memory=False)

    if Shuffle:
        Feature_Data = Feature_Data.sample(frac=1)

    Data_X = Feature_Data.iloc[:, :-1]
    Data_Y = Feature_Data.iloc[:, -1] - 1

    Feature_Data_index = Feature_Data.index.tolist()

    Adj_ = feature_to_adj(Data_X.values, k_nearest_neighobrs=k_nearest_neighobrs,
                          rm_common_neighbors=rm_common_neighbors)

    Adj = torch.from_numpy(np.array(Adj_, copy=True)).float()
    Data_X = torch.from_numpy(Data_X.to_numpy(copy=True)).float()
    Data_Y = torch.from_numpy(Data_Y.to_numpy(copy=True)).long()

    Mask_list = Get_Mask(Data_X, Data_Y, Device, n_splits=10, train_size=train_size, random_seed=Random_seed)
    train_num = [Mask_[0].int().sum().item() for Mask_ in Mask_list]
    test_num = [Mask_[1].int().sum().item() for Mask_ in Mask_list]

    in_channels = Data_X.shape[1]
    Class_num = len(torch.unique(Data_Y))

    assert Class_num == len(Class_names_list)

    Col_to_index_dict = {col: idx for idx, col in enumerate(Feature_Data.columns)}
    Feature_index_list = []
    Modal_Name_list = []
    for key, value in Feature_dict.items():
        Modal_Name_list.append(key)
        Feature_index_list.append([Col_to_index_dict[col] for col in value])

    Modal_Old_name = ['UCBERKELEYAV45', 'UCSFFSX', 'UPENNBIOMK9', 'Genetic_DATA']
    Modal_New_name = ['PET', 'MRI', 'CSF', 'PHS']
    Modal_name_replace_dict = dict(zip(Modal_Old_name, Modal_New_name))
    Modal_Name_list = [Modal_name_replace_dict.get(item, item) for item in Modal_Name_list]

    label_weight = count_elements(Data_Y.cpu().numpy())
    label_weight = label_weight.float()

    label_weight = label_weight.to(Device)
    Data_X = Data_X.to(Device)
    Data_Y = Data_Y.to(Device)
    Adj = Adj.to(Device)

    DATASET_Dict = {'Index': Feature_Data_index,
                    'Modal_Name': Modal_Name_list,
                    'Modal_Index': Feature_index_list,
                    'In_Channels': in_channels,
                    'Class_Num': Class_num,
                    'Adj': Adj,
                    'Feature_Num': Data_X.shape[1],
                    'Sample_Num': Data_X.shape[0],
                    'Class_Names': Class_names_list,
                    'Label_Weight': label_weight}

    DATASET_DATA = {'Feature': Data_X,
                    'Label': Data_Y,
                    'Mask': Mask_list,
                    'Train_Num': train_num,
                    'Test_Num': test_num}

    return DATASET_Dict, DATASET_DATA


def Get_Mask(Data_X, Data_Y, Device, n_splits=10, train_size=0.9, random_seed=0):
    Mask_list = []
    if train_size == 0:
        skf = StratifiedKFold(n_splits=n_splits, random_state=random_seed, shuffle=True)
        print('Using StratifiedKFold 10 folds for evaluation.')
    else:
        skf = StratifiedShuffleSplit(n_splits=n_splits, train_size=train_size, random_state=random_seed)
        print(f'Using StratifiedShuffleSplit {n_splits} folds with train size {train_size} for evaluation.')
    for train_index, test_index in skf.split(Data_X, Data_Y):
        train_mask = torch.zeros(Data_X.shape[0], dtype=torch.bool)
        train_mask[train_index] = True
        test_mask = ~train_mask
        Mask_list.append([train_mask.to(Device), test_mask.to(Device)])
    return Mask_list


def count_elements(matrix):
    unique_elements, counts = np.unique(matrix, return_counts=True)
    result = dict(zip(unique_elements, counts))
    sorted_dict = dict(sorted(result.items()))
    value_list = list(sorted_dict.values())
    # label_weight = torch.from_numpy(np.array(value_list) / matrix.shape[0])
    label_weight = torch.from_numpy((matrix.shape[0] - np.array(value_list)) / matrix.shape[0])

    return label_weight
