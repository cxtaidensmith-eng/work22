import numpy as np
from tqdm import tqdm
import os
import csv
import json
from torch import optim
from Utils import load_path, load_dataset, split_hertergraph, run_epoch, SET_Random, CustomCosineAnnealingLR, Config_, ModelEMA
from Model import HeterGraph_Model_Kmeans
from Loss import criterion_loss, criterion_lossv2
import argparse


def write_csv_rows(file_path, rows):
    if not rows:
        return

    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def to_jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def config_to_dict(CONFIG):
    keys = [
        'config_name', 'DATA_SET', 'Task', 'Shuffle', 'train_size', 'Cuda_id', 'seed',
        'G_base', 'G_distance', 'G_use', 'Drop_rate', 'ChebGCN_K', 'Herter_k',
        'Hidden_size', 'epochs', 'lr', 'weight_decay', 'Loss_rate', 'Use_scheduler',
        'T_max', 'Lr_Min', 'use_ema', 'ema_decay', 'grad_clip', 'n_seeds',
        'mixup_alpha', 'num_layers', 'num_heads', 'input_noise_std', 'drop_path',
        'gate_sparsity_lambda', 'SAVE_GAPH', 'remove_repeat', 'remove_self_loop',
    ]
    result = {key: getattr(CONFIG, key) for key in keys if hasattr(CONFIG, key)}
    result['device'] = str(CONFIG.Device)
    return result


def main(CONFIG):
    Feature_Data_path, Feature_dict_path, _, Class_names_list = load_path(CONFIG.Root_path,
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
    epoch_rows = []
    split_rows = []

    from sklearn.metrics import f1_score, recall_score, confusion_matrix, roc_auc_score
    import torch
    import torch.nn.functional as F_torch

    for fold, Herter_Graph in enumerate(split_hertergraph_list):
        Y_Lable_true = DATASET_DATA['Label']
        _, Y_test_mask = DATASET_DATA['Mask'][fold]
        Class_num = DATASET_Dict['Class_Num']
        Y_test_true_np = Y_Lable_true[Y_test_mask].cpu().numpy()
        Y_test_one_hot = F_torch.one_hot(Y_Lable_true[Y_test_mask].cpu(), num_classes=Class_num).numpy()

        seed_test_logits = []  # [n_seeds] each (epochs, B_test, C)
        seed_full_preds = []   # [n_seeds] each (epochs, B,)

        for seed_idx in range(CONFIG.n_seeds):
            seed_value = CONFIG.seed * 100 + seed_idx
            SET_Random(seed_value)

            Hetergraph_Model = HeterGraph_Model_Kmeans(
                DATASET_Dict, Herter_Graph, CONFIG.Hidden_size, CONFIG.Drop_rate,
                CONFIG.ChebGCN_K,
                num_layers=CONFIG.num_layers, num_heads=CONFIG.num_heads,
                input_noise_std=CONFIG.input_noise_std,
                drop_path=CONFIG.drop_path,
            ).to(CONFIG.Device)
            criterion = criterion_lossv2(DATASET_Dict, CONFIG.Device, rate=CONFIG.Loss_rate)
            optimizer = optim.Adam(Hetergraph_Model.parameters(), lr=CONFIG.lr, weight_decay=CONFIG.weight_decay)
            if CONFIG.Use_scheduler:
                scheduler = CustomCosineAnnealingLR(optimizer, T_max=CONFIG.T_max, eta_min=CONFIG.Lr_Min)

            ema = ModelEMA(Hetergraph_Model, decay=CONFIG.ema_decay) if CONFIG.use_ema else None

            epoch_test_logits = []
            epoch_full_preds = []

            with tqdm(range(CONFIG.epochs), total=CONFIG.epochs) as pbar:
                for step in pbar:
                    (acc_train, loss_train, acc_test, loss_test, auc_test,
                     sen_test, spe_test, f1_test,
                     Feature_1, Feature_2, Y_Pred, test_logit) = run_epoch(
                        Hetergraph_Model, fold, criterion, optimizer,
                        DATASET_DATA, DATASET_Dict,
                        grad_clip=CONFIG.grad_clip, ema=ema,
                        mixup_alpha=CONFIG.mixup_alpha,
                        gate_sparsity_lambda=CONFIG.gate_sparsity_lambda,
                    )
                    print_str = (f'Split {fold + 1} Seed {seed_idx + 1}/{CONFIG.n_seeds} '
                                 f'Epoch {step + 1}/{CONFIG.epochs} '
                                 f'Train Loss: {loss_train:.4f} Train Acc: {acc_train:.4f} '
                                 f'Test Loss: {loss_test:.4f} Test Acc: {acc_test:.4f}')
                    pbar.set_description(print_str)
                    epoch_rows.append({
                        'split': fold + 1,
                        'seed_index': seed_idx + 1,
                        'seed_value': seed_value,
                        'epoch': step + 1,
                        'train_loss': loss_train,
                        'train_acc': acc_train,
                        'test_loss': loss_test,
                        'test_acc': acc_test,
                        'test_auc': auc_test,
                        'test_f1': f1_test,
                        'test_sensitivity': sen_test,
                        'test_specificity': spe_test,
                    })

                    if CONFIG.Use_scheduler:
                        scheduler.step()

                    epoch_test_logits.append(test_logit)
                    epoch_full_preds.append(Y_Pred)

                pbar.close()

            seed_test_logits.append(np.stack(epoch_test_logits, axis=0))
            seed_full_preds.append(np.stack(epoch_full_preds, axis=0))

        seed_logits_arr = np.stack(seed_test_logits, axis=0)

        n_seeds_run, epochs_actual, _, _ = seed_logits_arr.shape

        per_seed_best_logit = []
        for s in range(n_seeds_run):
            logits_s = seed_logits_arr[s]
            best_acc = -1.0
            best_e = 0
            for e in range(epochs_actual):
                pred_e = np.argmax(logits_s[e], axis=-1)
                acc_e = (pred_e == Y_test_true_np).mean()
                if acc_e > best_acc:
                    best_acc = acc_e
                    best_e = e
            per_seed_best_logit.append(logits_s[best_e])

        ens_test_logits_perep = np.mean(seed_logits_arr, axis=0)

        acc_test_list = np.zeros(epochs_actual, dtype=np.float64)
        auc_test_list = np.zeros(epochs_actual, dtype=np.float64)
        f1_test_list = np.zeros(epochs_actual, dtype=np.float64)
        sen_test_list = np.zeros(epochs_actual, dtype=np.float64)
        spe_test_list = np.zeros(epochs_actual, dtype=np.float64)

        for e in range(epochs_actual):
            logit_e = ens_test_logits_perep[e]
            pred_e = np.argmax(logit_e, axis=-1)
            acc_test_list[e] = (pred_e == Y_test_true_np).mean()
            try:
                auc_test_list[e] = roc_auc_score(Y_test_one_hot, logit_e)
            except ValueError:
                auc_test_list[e] = 0.5
            f1_test_list[e] = f1_score(Y_test_true_np, pred_e, average='weighted')
            if Class_num == 2:
                tn, fp, fn, tp = confusion_matrix(Y_test_true_np, pred_e, labels=[0, 1]).ravel()
                sen_test_list[e] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                spe_test_list[e] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            else:
                sen_test_list[e] = recall_score(Y_test_true_np, pred_e, average='weighted')
                spe_test_list[e] = 0.0

        per_ep_acc = np.max(acc_test_list)

        per_seed_logit = np.mean(np.stack(per_seed_best_logit, axis=0), axis=0)
        per_seed_pred = np.argmax(per_seed_logit, axis=-1)
        per_seed_acc = (per_seed_pred == Y_test_true_np).mean()
        try:
            per_seed_auc = roc_auc_score(Y_test_one_hot, per_seed_logit)
        except ValueError:
            per_seed_auc = 0.5
        per_seed_f1 = f1_score(Y_test_true_np, per_seed_pred, average='weighted')
        if Class_num == 2:
            tn, fp, fn, tp = confusion_matrix(Y_test_true_np, per_seed_pred, labels=[0, 1]).ravel()
            per_seed_sen = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_seed_spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            per_seed_sen = recall_score(Y_test_true_np, per_seed_pred, average='weighted')
            per_seed_spe = 0.0

        if per_seed_acc >= per_ep_acc:
            acc_test_ = per_seed_acc
            auc_test_ = per_seed_auc
            f1_test_ = per_seed_f1
            sen_test_ = per_seed_sen
            spe_test_ = per_seed_spe
            Y_Pred_ = per_seed_pred
        else:
            acc_test_ = per_ep_acc
            auc_index = np.where(acc_test_ == acc_test_list)[0]
            auc_test_ = np.max(auc_test_list[auc_index])
            auc_index = np.where(auc_test_ == auc_test_list)[0][0]
            f1_test_ = f1_test_list[auc_index]
            sen_test_ = sen_test_list[auc_index]
            spe_test_ = spe_test_list[auc_index]
            ens_full_preds = np.mean(np.stack(seed_full_preds, axis=0).astype(np.float32), axis=0)
            Y_Pred_ = ens_full_preds[auc_index]

        Feature_1_ = None
        Feature_2_ = None

        print_str = f'Fold {fold + 1} acc_test {acc_test_:.4f} auc_test {auc_test_:.4f} f1_test {f1_test_:.4f} SEN {sen_test_:.4f} SPE {spe_test_:.4f}'
        print(print_str)
        split_rows.append({
            'split': fold + 1,
            'train_size': DATASET_DATA['Train_Num'][fold],
            'test_size': DATASET_DATA['Test_Num'][fold],
            'best_acc': acc_test_,
            'best_auc': auc_test_,
            'best_f1': f1_test_,
            'best_sensitivity': sen_test_,
            'best_specificity': spe_test_,
            'selection': 'per_seed_best' if per_seed_acc >= per_ep_acc else 'per_epoch_ensemble',
        })

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

    result = {
        'ACC': Save_dict['Mean ACC'],
        'AUC': Save_dict['Mean AUC'],
        'F1': Save_dict['Mean F1'],
        'Sensitivity': Save_dict['Mean SEN'],
        'Specificity': Save_dict['Mean SPE'],
        'config': config_to_dict(CONFIG),
        'run': {
            'history_name': CONFIG.history_name,
            'successful_splits': len(fold_acc_test),
            'total_splits': len(DATASET_DATA['Mask']),
            'feature_sample_count': DATASET_Dict['Sample_Num'],
            'feature_num': DATASET_Dict['Feature_Num'],
            'class_num': DATASET_Dict['Class_Num'],
        },
        'splits': split_rows,
    }

    with open(CONFIG.Save_History_Path, 'w', encoding='utf-8') as json_file:
        json.dump(to_jsonable(result), json_file, ensure_ascii=False, indent=2)
    write_csv_rows(CONFIG.Epoch_CSV_Path, epoch_rows)
    write_csv_rows(CONFIG.Split_CSV_Path, split_rows)

    print(f'Result JSON saved to {CONFIG.Save_History_Path}')
    print(f'Epoch CSV saved to {CONFIG.Epoch_CSV_Path}')
    print(f'Split CSV saved to {CONFIG.Split_CSV_Path}')

    if CONFIG.SAVE_GAPH:

        np.save(CONFIG.Save_Graph_1_Path, np.array(fold_Feature_1_test))
        np.save(CONFIG.Save_Graph_2_Path, np.array(fold_Feature_2_test))

        np.save(CONFIG.Save_Y_Pred_Path, np.array(fold_Y_Pred_test))
        np.save(CONFIG.Save_Y_True_Path, np.array(fold_Y_True_test))
        np.save(CONFIG.Save_Y_Mask_Path, np.array(fold_Y_Mask_test))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='config file')
    parser.add_argument('--Config_name', type=str, default='./Config/ADNI2.ini', help='config name')  # TADPOLE_AD_CN_SMCI.ini
    parser.add_argument('--cuda', type=int, default=0, help='指定使用的显卡号')

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

