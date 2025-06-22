import numpy as np
from tqdm import tqdm
import os
from torch import optim
from Utils import load_path, load_dataset, split_hertergraph, run_epoch, SET_Random, CustomCosineAnnealingLR, Config_
from Model import HeterGraph_Model_Kmeans
from Loss import criterion_loss, criterion_lossv2
import argparse


def main(CONFIG):
    Feature_Data_path, Feature_dict_path, Save_History_path, Class_names_list = load_path(CONFIG.Root_path,
                                                                                          CONFIG.DATA_SET, CONFIG.Task)
    DATASET_Dict, DATASET_DATA = load_dataset(Feature_Data_path, Feature_dict_path, CONFIG.Device, Class_names_list,
                                              CONFIG.Shuffle,
                                              CONFIG.seed,
                                              train_size=CONFIG.train_size)
    split_hertergraph_list = split_hertergraph(DATASET_Dict, DATASET_DATA, CONFIG, K=CONFIG.Herter_k,
                                               remove_self_loop=CONFIG.remove_self_loop,
                                               remove_repeat=CONFIG.remove_repeat)

    # for split_index, split_data in enumerate(split_hertergraph_list):
    #     for lable_index, lable_graph_data in enumerate(split_data):
    #         for k_index, K_graph in enumerate(lable_graph_data):
    #             graph_dict, graph_edge = K_graph
    #             print(graph_dict, graph_edge)

    fold_acc_test = []
    fold_auc_test = []
    fold_f1_test = []
    fold_sen_test = []
    fold_spe_test = []

    fold_Feature_1_test = []
    fold_Feature_2_test = []

    fold_Y_Pred_test = []
    fold_Y_True_test = []
    fold_Y_Mask_test = []

    Save_dict = {}

    for fold, Herter_Graph in enumerate(split_hertergraph_list):

        Hetergraph_Model = HeterGraph_Model_Kmeans(DATASET_Dict, Herter_Graph, CONFIG.Hidden_size, CONFIG.Drop_rate,
                                                   CONFIG.ChebGCN_K).to(CONFIG.Device)
        criterion = criterion_lossv2(DATASET_Dict, CONFIG.Device, rate=CONFIG.Loss_rate)
        optimizer = optim.Adam(Hetergraph_Model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=250, eta_min=Lr_Min)
        if CONFIG.Use_scheduler:
            scheduler = CustomCosineAnnealingLR(optimizer, T_max=CONFIG.T_max, eta_min=CONFIG.Lr_Min)

        acc_test_list = []
        auc_test_list = []
        f1_test_list = []
        sen_test_list = []
        spe_test_list = []

        Feature_1_test_list = []
        Feature_2_test_list = []

        Y_Pred_test_list = []

        # for i in range(CONFIG.epochs):
        #     acc_train, loss_train, acc_test, loss_test, auc_test = run_epoch(Hetergraph_Model, fold, criterion, optimizer, DATASET_DATA, DATASET_Dict)
        #     print(
        #         f'Epoch {i + 1}/{CONFIG.epochs} Train Loss: {loss_train:.4f} Train Acc: {acc_train:.4f} Test Loss: {loss_test:.4f} Test Acc: {acc_test:.4f}')

        #     if CONFIG.Use_scheduler:
        #         scheduler.step()

        #     acc_test_list.append(acc_test)
        #     auc_test_list.append(auc_test)

        with tqdm(range(CONFIG.epochs), total=CONFIG.epochs) as pbar:
            for step in pbar:
                acc_train, loss_train, acc_test, loss_test, auc_test, sen_test, spe_test, f1_test, Feature_1, Feature_2, Y_Pred = run_epoch(Hetergraph_Model,
                                                                                                     fold, criterion,
                                                                                                     optimizer,
                                                                                                     DATASET_DATA,
                                                                                                     DATASET_Dict)
                print_str = f'Split {fold + 1} Epoch {step + 1}/{CONFIG.epochs} Train Loss: {loss_train:.4f} Train Acc: {acc_train:.4f} Test Loss: {loss_test:.4f} Test Acc: {acc_test:.4f}'
                pbar.set_description(print_str)
                CONFIG.logger.info(print_str)

                if CONFIG.Use_scheduler:
                    scheduler.step()

                acc_test_list.append(acc_test)
                auc_test_list.append(auc_test)
                f1_test_list.append(f1_test)
                sen_test_list.append(sen_test)
                spe_test_list.append(spe_test)

                Feature_1_test_list.append(Feature_1)
                Feature_2_test_list.append(Feature_2)

                Y_Pred_test_list.append(Y_Pred)

            pbar.close()

        Y_Lable_true = DATASET_DATA['Label']
        _, Y_test_mask = DATASET_DATA['Mask'][fold]

        acc_test_ = np.max(acc_test_list)
        auc_index = np.where(acc_test_ == acc_test_list)[0]
        auc_test_ = np.max(np.array(auc_test_list)[auc_index])
        auc_index = np.where(auc_test_ == auc_test_list)[0][0]

        f1_test_ = np.array(f1_test_list)[auc_index]
        sen_test_ = np.array(sen_test_list)[auc_index]
        spe_test_ = np.array(spe_test_list)[auc_index]

        Feature_1_ = np.array(Feature_1_test_list)[auc_index]
        Feature_2_ = np.array(Feature_2_test_list)[auc_index]

        Y_Pred_ = np.array(Y_Pred_test_list)[auc_index]

        print_str = f'Fold {fold + 1} acc_test {acc_test_:.4f} auc_test {auc_test_:.4f} f1_test {f1_test_:.4f} SEN {sen_test_:.4f} SPE {spe_test_:.4f}'
        print(print_str)
        CONFIG.logger.info(print_str)

        fold_acc_test.append(acc_test_)
        fold_auc_test.append(auc_test_)
        fold_f1_test.append(f1_test_)
        fold_sen_test.append(sen_test_)
        fold_spe_test.append(spe_test_)

        fold_Feature_1_test.append(Feature_1_)
        fold_Feature_2_test.append(Feature_2_)

        fold_Y_Pred_test.append(Y_Pred_)
        fold_Y_True_test.append(Y_Lable_true.cpu().numpy())
        fold_Y_Mask_test.append(Y_test_mask.cpu().numpy())

        Save_dict[f'fold: {fold} ACC'] = acc_test_list
        Save_dict[f'fold: {fold} AUC'] = auc_test_list
        Save_dict[f'fold: {fold} F1'] = f1_test_list
        Save_dict[f'fold: {fold} SEN'] = sen_test_list
        Save_dict[f'fold: {fold} SPE'] = spe_test_list

    Save_dict['fold: best ACC'] = fold_acc_test
    Save_dict['fold: best AUC'] = fold_auc_test
    Save_dict['fold: best F1'] = fold_f1_test
    Save_dict['fold: best SEN'] = fold_sen_test
    Save_dict['fold: best SPE'] = fold_spe_test

    acc_mean, acc_std = np.mean(fold_acc_test), np.std(fold_acc_test)
    auc_mean, auc_std = np.mean(fold_auc_test), np.std(fold_auc_test)
    f1_mean, f1_std = np.mean(fold_f1_test), np.std(fold_f1_test)
    sen_mean, sen_std = np.mean(fold_sen_test), np.std(fold_sen_test)
    spe_mean, spe_std = np.mean(fold_spe_test), np.std(fold_spe_test)

    Save_dict['Mean ACC'] = {'Mean': acc_mean, 'Std': acc_std}
    Save_dict['Mean AUC'] = {'Mean': auc_mean, 'Std': auc_std}
    Save_dict['Mean F1'] = {'Mean': f1_mean, 'Std': f1_std}
    Save_dict['Mean SEN'] = {'Mean': sen_mean, 'Std': sen_std}
    Save_dict['Mean SPE'] = {'Mean': spe_mean, 'Std': spe_std}

    print_str = (f'Fold ACC Mean {acc_mean:.4f} Std {acc_std:.4f} | AUC Mean {auc_mean:.4f} Std {auc_std:.4f} \n'
                 f'F1 Mean {f1_mean:.4f} Std {f1_std:.4f} \n'
                 f'SEN Mean {sen_mean:.4f} Std {sen_std:.4f} | SPE Mean {spe_mean:.4f} Std {spe_std:.4f} ')
    print(print_str)
    CONFIG.logger.info(print_str)

    np.save(CONFIG.Save_History_Path, Save_dict)

    if CONFIG.SAVE_GAPH:

        np.save(CONFIG.Save_Graph_1_Path, np.array(fold_Feature_1_test))
        np.save(CONFIG.Save_Graph_2_Path, np.array(fold_Feature_2_test))

        np.save(CONFIG.Save_Y_Pred_Path, np.array(fold_Y_Pred_test))
        np.save(CONFIG.Save_Y_True_Path, np.array(fold_Y_True_test))
        np.save(CONFIG.Save_Y_Mask_Path, np.array(fold_Y_Mask_test))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--Config_name', type=str, default='./Config/ADNI2.ini', help='config name')  # TADPOLE_AD_CN_SMCI.ini
    parser.add_argument('--cuda', type=int, default=7, help='指定使用的显卡号')

    # 解析命令行参数
    args = parser.parse_args()
    Config_name = args.Config_name
    cuda_device = args.cuda

    # Config_name = 'TADPOLE_AD_CN_SMCI.ini'  # TADPOLE_AD_CN_SMCI.ini   TADPOLE_SMCI_PMCI.ini  'config_v3.ini
    
    try:
        Root_path = os.path.dirname(os.path.abspath(__file__))
    except:
        Root_path = os.getcwd()

    CONFIG = Config_(Root_path, Config_name, cuda_device)

    SET_Random(CONFIG.seed)
    main(CONFIG)

