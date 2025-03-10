#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import torch
import os
import shutil
from biopandas.pdb import PandasPdb
import pandas as pd
import subprocess
import glob
from scipy.spatial import distance_matrix
from collections import defaultdict
from dataset_util.toughm1_dataset_for_pyg_8dim import TOUGH_M1_pair_dataset, TOUGH_M1_pair_dataset_collate_fn
from models.simsegpocket_models_pyg import feature_model_v2
from models.loss_modules import feature_loss_func
from time import time
from datetime import datetime, UTC, timedelta
from tqdm import tqdm
from dataset_util.toughm1_utils import ToughM1
from tqdm import tqdm
#from torch.utils.data import DataLoader
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import argparse
import multiprocessing
import math
import logging
from sklearn.metrics import roc_auc_score
from python_exe_command_get import log_command
import yaml
import importlib
import traceback

logger = logging.getLogger(__name__)
args = None
log_folder = None

def import_module_from_file(file_path):
    logger.info(f"import module from {file_path}")
    module_name = file_path.split('/')[-1].split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def get_model_object(log_folder,model_file,model_name):
    # モジュールをインポート
    module = import_module_from_file(os.path.join(log_folder,model_file))

    # 特定の関数またはクラスを取得
    if hasattr(module, model_name):
        model = getattr(module, model_name)
        return model
    else:
        raise AttributeError(f"Module does not have an attribute named '{model_name}'")
    

def torch_fix_seed(seed=0):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True

# adopted from pointnet++ pointnet2_utils.py


def Rotation_xyz(coords):
    # Rotation
    rng = np.random.default_rng()
    theta_x = rng.uniform(0,2*math.pi)
    theta_y = rng.uniform(0,2*math.pi)
    theta_z = rng.uniform(0,2*math.pi)
    rot_x = torch.tensor([[ 1,                 0,                  0],
                          [ 0, math.cos(theta_x), -math.sin(theta_x)],
                          [ 0, math.sin(theta_x),  math.cos(theta_x)]])

    rot_y = torch.tensor([[ math.cos(theta_y), 0,  math.sin(theta_y)],
                          [                 0, 1,                  0],
                          [-math.sin(theta_y), 0, math.cos(theta_y)]])

    rot_z = torch.tensor([[ math.cos(theta_z), -math.sin(theta_z), 0],
                          [ math.sin(theta_z),  math.cos(theta_z), 0],
                          [                 0,                  0, 1]])

    rot_matrix = torch.mm(rot_z,torch.mm(rot_y,rot_x))
    rot_coords = torch.mm(rot_matrix, coords.squeeze().T).T
    #print(rot_coords.size())
    translated_rot_coords = rot_coords + torch.randn(1,3).repeat(rot_coords.size(0),1)

    
    return translated_rot_coords

class Objective:
    def __init__(self, train_db, test_db):
        self.train_db = train_db
        self.test_db = test_db
        #self.train_db = random.sample(self.train_db,len(self.train_db)//5)
        #self.test_db = random.sample(self.test_db,len(self.test_db)//5)

        torch.save({
            "train_db":train_db,"test_db":test_db
        },f"{log_folder}/database_to_use.pickle")
        
    def __call__(self):
        torch.set_num_threads(8)
        gpu_id = args.cuda
        device = f"cuda:{gpu_id}"
        #device = 'cpu'
        logger.info(f'Using device: '+device)
        
        # ハイパラ
        
        # log_folder関連
        model_file = "models/simsegpocket_models_pyg.py"
        #model_name = "feature_model_v2"
        model_name = getattr(args, "feature_model_name", "feature_model_v2")
        loss_module_file = "models/loss_modules.py"
        loss_module_name = "feature_loss_func"
        model_state_dict_to_load = getattr(args, "model_to_load", None)
        
        # feature_model関連
        num_SAModule = getattr(args, "num_SAModule", 4)
        dropout = 0.25
        method = getattr(args, "method", "segmented")
        if method == "segmented_add":
            in_channels = 9
        else:
            in_channels = 8
        out_channels = 128
        last_act = args.last_act
        customFPS = getattr(args, "custom_fps", False)
        ratio = getattr(args, "ratio", 0.25)
        customNet = getattr(args, "custom_net", False)
        
        # dataset
        data_root = args.toughm1_data_root
        recalculate = args.channel_recalculate
        pocket_segmented = getattr(args, "pocket_segmented", True)
        
        # dataloader
        batch_size = 2 ** args.batch_size
        num_workers = args.num_workers
        
        # 損失関数
        positive_loss_type = args.positive_loss
        negative_loss_type = args.negative_loss
        reduction = "mean"
        margin = 1
        rotation_invariance_loss = getattr(args, "rotInvLoss", False)
        rotation_invariance_loss_version = getattr(args, "rotation_invariance_loss_version", 2)
        
        # other
        lr = 2.5e-05
        num_epoch = args.num_epoch
        method = getattr(args, "method", "segmented")
        if "segmented" in method:
            pocket_segmented = True
        else:
            pocket_segmented = False
        p2rank_prediction_apply = getattr(args, "p2rank_prediction_apply", True)
        
        hyperparameters = {
            "logs" : {
                "log_folder" : log_folder,
                "data_file" : "database_to_use.pickle",
                "model_file" : model_file,
                "model_name" : model_name,
                "loss_module_file" : loss_module_file,
                "loss_module_name" : loss_module_name,
                "model_state_dict" : "model.pth",
                "model_state_dict_first": model_state_dict_to_load
            },
            "model_feature_parameters" : {
                "num_SAModule":num_SAModule,
                "dropout":dropout,
                "in_channels":in_channels,
                "out_channels":out_channels,
                "last_act":last_act,
                "custom_fps":customFPS,
                "ratio":ratio,
                "custom_net":customNet
            },
            "dataset_settings" : {
                "data_root": data_root,
                "valid_data": args.valid_data,
                "fold_nr": args.fold_nr,
                "recalculate": recalculate,
                "moleculekit_version": args.moleculekit_version,
                "include_f1": args.include_f1,
                "p2rank_prediction_apply": p2rank_prediction_apply,
                "p2rank_col": args.p2rank_col,
                "pocket_segmented": pocket_segmented
            },
            "dataloader_settings" : {
                "batch_size": batch_size,
                "num_workers": num_workers
            },
            "lossfunc_settings" : {
                "positive_loss_type" : positive_loss_type,
                "negative_loss_type" : negative_loss_type,
                "reduction": reduction,
                "margin": margin,
                "rotation_invariance_loss":rotation_invariance_loss,
                "rotation_invariance_loss_version":rotation_invariance_loss_version
            },
            "other_settings" : {
                "lr":lr,
                "num_epoch":num_epoch,
                "method":method
            }
        }
        
        with open(f'{log_folder}/feature_hyperparameters.yml', 'w') as file:
            yaml.dump(hyperparameters, file, default_flow_style=False)
        
        try:
            fold_nr = args.valid_data if args.valid_data is not None else hyperparameters["dataset_settings"]["fold_nr"]
            test_data = torch.load(args.toughm1_data_root.replace("/TOUGH-M1/",f"/train_dataset_{fold_nr}.pt")) #non-recalculate Dataset
            assert test_data["data_root"] == args.toughm1_data_root
            pair_train_dataset = test_data["dataset"]
            pair_train_dataset.set_col(args.p2rank_col)
            if recalculate==True:
                pair_train_dataset.set_recalculate(recalculate, device)
        except Exception as e:
            logger.info(f"{e}, construct TOUGH_M1_pair_dataset")
            pair_train_dataset = TOUGH_M1_pair_dataset(self.train_db, data_root=args.toughm1_data_root, recalculate=args.channel_recalculate, col=args.p2rank_col)
            
        if self.test_db is not None:
            try:
                fold_nr = args.valid_data if args.valid_data is not None else hyperparameters["dataset_settings"]["fold_nr"]
                test_data = torch.load(args.toughm1_data_root.replace("/TOUGH-M1/",f"/train_dataset_{fold_nr}.pt")) #non-recalculate Dataset
                assert test_data["data_root"] == args.toughm1_data_root
                pair_test_dataset = test_data["test_dataset"]
                pair_test_dataset.set_col(args.p2rank_col)
                if recalculate==True:
                    pair_test_dataset.set_recalculate(recalculate, device)
            except Exception as e:
                logger.info(f"{e}, construct TOUGH_M1_pair_dataset")
                pair_test_dataset = TOUGH_M1_pair_dataset(self.test_db, data_root=args.toughm1_data_root, recalculate=args.channel_recalculate, col=args.p2rank_col)
        else:
            pair_test_dataset = None
        
        #torch.save({
        #    "dataset":pair_train_dataset, "test_dataset":pair_test_dataset, "data_root":data_root, "recalculate":recalculate
        #},args.toughm1_data_root.replace("/TOUGH-M1/",f"/train_dataset_{fold_nr}.pt"))
        
        device = torch.device(device)
        
        # モデルの初期化
        # モデルの初期化
        # from models.simsegpocket_models_pyg import feature_model_v2
        feature_model_v2 = get_model_object(log_folder, model_file, model_name)
        model_feature = feature_model_v2(out_channels=out_channels,in_channels=in_channels,num_SAModule=num_SAModule, dropout=dropout, last_act=last_act, custom_fps=customFPS, custom_net=customNet).to(device) #num_classes, in_channels
        if model_state_dict_to_load is not None:
            model_feature.load_state_dict(torch.load(model_state_dict_to_load))
            new_path = shutil.copy(model_state_dict_to_load,os.path.join(log_folder,"model_to_load.pth"))
        # from models.loss_modules import feature_loss_func
        feature_loss_func = get_model_object(log_folder, loss_module_file, loss_module_name)
        loss_func = feature_loss_func(margin=margin,positive_loss_type=positive_loss_type,negative_loss_type=negative_loss_type,reduction=reduction,rotation_invariance_loss=rotation_invariance_loss, rotation_invariance_loss_version=rotation_invariance_loss_version)
        ## pair datasetの検証
        logger.debug(data_root)
        
        pair_train_dataloader = DataLoader(pair_train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=TOUGH_M1_pair_dataset_collate_fn)
        if pair_test_dataset is not None:
            pair_test_dataloader = DataLoader(pair_test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=TOUGH_M1_pair_dataset_collate_fn)
        else:
            pair_test_dataloader = None
        optimizer = torch.optim.Adam(list(model_feature.parameters()), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min')

        start = time()
        train_loss_logs = []
        train_rot_inv_loss_logs = []
        train_auc_logs = []
        test_loss_logs = []
        test_rot_inv_loss_logs = []
        test_auc_logs = []
        for epoch in range(num_epoch):
            logger.info(f"Epoch {epoch} start / {num_epoch}")
            # train mode
            model_feature.train()
            loss_per_epoch = 0
            rot_inv_loss_per_epoch = 0
            diff_list = []
            label_list = []
            count = 0
            for i,pair_data in enumerate(pair_train_dataloader):
                optimizer.zero_grad()
                logger.debug(f"pair_data:{pair_data}")
                logger.debug(f"Iter {i} start")
                features = []
                pocket_ids = []
                start_data = time()
                
                # batch_sub がポケットの番号を示していることを予め確認
                #logger.debug(f"batch{pair_data.batch},batch_sub:{pair_data.batch_sub},label:{pair_data.label}")
                assert len(torch.unique(pair_data.batch))==len(pair_data.label)
                
                logger.debug(f"coords:{pair_data.pos.size()},atom_feature:{pair_data.x.size()},pair_data.batch:{pair_data.batch},pair_data.batch_sub:{pair_data.batch_sub}")
                logger.debug(f"data picked {time()-start_data}s")
                coords = Rotation_xyz(pair_data.pos).to(device)
                if method=="segmented_add":
                    atom_feature = torch.cat([pair_data.x, pair_data.pocket_flag.unsqueeze(1)], dim=1)
                    atom_feature = atom_feature.to(device)
                elif method=="segmented_multiply":
                    atom_feature = pair_data.x * pair_data.pocket_flag.unsqueeze(1)
                    atom_feature = atom_feature.to(device)
                else:    
                    atom_feature = pair_data.x.to(device)
                sub_batch = pair_data.batch_sub.to(device)
                logger.debug(f"coords.size()={coords.size()},atom_feature.size()={atom_feature.size()}")
                #pocket_feature_overall = torch.cat((coords,atom_feature),dim=1).unsqueeze(0).permute(0,2,1)
                logger.debug(f"data send to device {time()-start_data}s")
                #logger.debug(f"pocket_feature_overall:{pocket_feature_overall.size()}")
                x_feature = model_feature(atom_feature,coords,sub_batch) # バッチサイズ×特徴量次元数のテンソルがでてくるはず
                logger.debug(f"model forward {time()-start_data}s, output.size():{x_feature.size()}")
                labels = pair_data.label.to(device)
                start_loss_calc = time()
                
                if rotation_invariance_loss:
                    coords2 = Rotation_xyz(pair_data.pos).to(device)
                    x_feature2 = model_feature(atom_feature,coords2,sub_batch)
                    euclidean_diff,rot_inv_loss,loss = loss_func((x_feature, x_feature2),labels)
                else:
                    euclidean_diff,loss = loss_func(x_feature,labels)
                
                loss.backward()
                optimizer.step()
                diff_list.append(euclidean_diff.cpu())
                label_list.append(labels.cpu())
                logger.debug(f"euclidean_diff:{euclidean_diff},{time()-start_loss_calc}s")
                logger.debug(f"train loss value:{loss.cpu().item()},{time()-start_loss_calc}s")
                logger.debug(f"iter {i+1} finished {time()-start}s")
                loss_per_epoch += loss.cpu().item()
                count += len(torch.unique(pair_data.batch))
                if rotation_invariance_loss:
                    rot_inv_loss_per_epoch += rot_inv_loss.cpu().item()
            loss_per_epoch /= count
            train_loss_logs.append(loss_per_epoch)
            rot_inv_loss_per_epoch /= count
            train_rot_inv_loss_logs.append(rot_inv_loss_per_epoch)
            diffs = torch.cat(diff_list).detach().numpy().copy()
            labels = torch.cat(label_list).detach().numpy().copy()
            train_auc_logs.append(roc_auc_score(labels,-diffs))
            logger.info(f"epoch {epoch} train loss value:{loss_per_epoch}, {time()-start}s")
            
            # test mode
            if pair_test_dataloader is not None:
                model_feature.eval()
                loss_per_epoch = 0
                diff_list = []
                label_list = []
                count = 0
                for i,pair_data in enumerate(pair_test_dataloader):
                    with torch.no_grad():
                        #optimizer.zero_grad()
                        logger.debug(f"pair_data:{pair_data}")
                        logger.debug(f"Iter {i} start")
                        features = []
                        pocket_ids = []
                        start_data = time()
                        
                        # batch_sub がポケットの番号を示していることを予め確認
                        logger.debug(f"batch{pair_data.batch},batch_sub:{pair_data.batch_sub},label:{pair_data.label}")
                        assert len(torch.unique(pair_data.batch))==len(pair_data.label)
                        
                        logger.debug(f"coords:{pair_data.pos.size()},atom_feature:{pair_data.x.size()},pair_data.batch:{pair_data.batch},pair_data.batch_sub:{pair_data.batch_sub}")
                        logger.debug(f"data picked {time()-start_data}s")
                        coords = Rotation_xyz(pair_data.pos).to(device)
                        if method=="segmented_add":
                            atom_feature = torch.cat([pair_data.x, pair_data.pocket_flag.unsqueeze(1)], dim=1)
                            atom_feature = atom_feature.to(device)
                        elif method=="segmented_multiply":
                            atom_feature = pair_data.x * pair_data.pocket_flag.unsqueeze(1)
                            atom_feature = atom_feature.to(device)
                        else:    
                            atom_feature = pair_data.x.to(device)
                        sub_batch = pair_data.batch_sub.to(device)
                        logger.debug(f"coords.size()={coords.size()},atom_feature.size()={atom_feature.size()}")
                        #pocket_feature_overall = torch.cat((coords,atom_feature),dim=1).unsqueeze(0).permute(0,2,1)
                        logger.debug(f"data send to device {time()-start_data}s")
                        #logger.debug(f"pocket_feature_overall:{pocket_feature_overall.size()}")
                        x_feature = model_feature(atom_feature,coords,sub_batch) # バッチサイズ×特徴量次元数のテンソルがでてくるはず
                        logger.debug(f"model forward {time()-start_data}s, output.size():{x_feature.size()}")
                        labels = pair_data.label.to(device)
                        start_loss_calc = time()
                        if rotation_invariance_loss:
                            coords2 = Rotation_xyz(pair_data.pos).to(device)
                            x_feature2 = model_feature(atom_feature,coords2,sub_batch)
                            euclidean_diff,rot_inv_loss,loss = loss_func((x_feature, x_feature2),labels)
                        else:
                            euclidean_diff,loss = loss_func(x_feature,labels)
                            
                        diff_list.append(euclidean_diff.cpu())
                        label_list.append(labels.cpu())
                        logger.debug(f"euclidean_diff:{euclidean_diff},{time()-start_loss_calc}s")
                        logger.debug(f"test loss value:{loss.cpu().item()},{time()-start_loss_calc}s")
                        #loss.backward()
                        logger.debug(f"iter {i+1} finished {time()-start}s")
                        #optimizer.step()
                        loss_per_epoch += loss.cpu().item()
                        count += len(torch.unique(pair_data.batch))
                        if rotation_invariance_loss:
                            rot_inv_loss_per_epoch += rot_inv_loss.cpu().item()
                if count > 0:
                    loss_per_epoch /= count
                    rot_inv_loss_per_epoch /= count
                    diffs = torch.cat(diff_list).detach().numpy().copy()
                    labels = torch.cat(label_list).detach().numpy().copy()
                    test_auc_logs.append(roc_auc_score(labels,-diffs))
                else:
                    loss_per_epoch = None
                    test_auc_logs.append(None)
                
                # test_per_epochがあまりにも変化なかったら学習率を下げる
                #scheduler.step(loss_per_epoch)
                logger.info(f"epoch {epoch} test loss value:{loss_per_epoch}, {time()-start}s")
                test_loss_logs.append(loss_per_epoch)
                test_rot_inv_loss_logs.append(rot_inv_loss_per_epoch)
                
                
                
            else:
                test_loss_logs.append(None)
                test_auc_logs.append(None)
            train_loss = pd.DataFrame({"train_loss":train_loss_logs,"train_rot_inv_loss":train_rot_inv_loss_logs,"train_auc":train_auc_logs,
                                       "test_loss":test_loss_logs,  "test_rot_inv_loss":test_rot_inv_loss_logs,  "test_auc":test_auc_logs})
            train_loss.to_csv(f"{log_folder}/loss_logs.csv",encoding='utf-8')
            
            if (epoch + 1) % 50 == 0:
                torch.save(model_feature.cpu().state_dict(), f"{log_folder}/model_{epoch+1}.pth")  
                model_feature = model_feature.to(device)
                
            torch.cuda.empty_cache()
        
        torch.save(model_feature.cpu().state_dict(), f"{log_folder}/model.pth")   
        return

if __name__ == '__main__':
    # logを保存するファイル
    parser = argparse.ArgumentParser()
    
    # parser.add_argumentで受け取る引数を追加していく
    parser.add_argument('-b', '--batch_size', type=int, default=10, help="batch size")
    parser.add_argument('-w', '--num_workers', type=int, default=8, help="cpu sum")
    parser.add_argument('-a', '--appendix', default="", help="appendix of file name")
    parser.add_argument('-c', '--cuda', default=0, help="cuda id")
    parser.add_argument('-e', '--num_epoch', type=int, default=100, help="epoch number")
    parser.add_argument('-d', '--debug_mode', type=bool, default=False, help="デバッグモードでやるかどうか")
    parser.add_argument('--channel_recalculate', type=str, default="False", help="DeeplyToughと同様にチャンネル値を計算し直すか")
    parser.add_argument('--toughm1_data_root', default="./TOUGH-M1/", help="TOUGH-M1データセットディレクトリの場所")
    parser.add_argument('--fold_nr', type=int, default=0)
    parser.add_argument('--positive_loss', default="normal",help="正のペアに対する損失関数")
    parser.add_argument('--negative_loss', default="relu_log",help="負のペアに対する損失関数")
    parser.add_argument('--valid_data', default="None", type=str, help="vertex,prospeccts,...")
    parser.add_argument("--include_f1", type=str, default="False")
    parser.add_argument('--last_act', type=str, default="relu", help="feature_model_v2の最終層にreluをかけるか否か")
    parser.add_argument("--moleculekit_version", type=str, default="v2")
    parser.add_argument("--custom_fps", type=str, default="False")
    parser.add_argument("--rotInvLoss",  type=str, default="False")
    parser.add_argument("--rotation_invariance_loss_version", type=int, default=2)
    parser.add_argument("--p2rank_prediction_apply", type=str, default="False")
    parser.add_argument("--pocket_segmented", type=str, default="False")
    parser.add_argument("--lr",type=float,default=2.5e-05)
    parser.add_argument("--feature_model_name",type=str,default="feature_model_v2")
    parser.add_argument("--database_path", type=str, default=None)
    parser.add_argument("--method", type=str, default="segmented", help="segmented|segmented_add|segmented_multiply")
    parser.add_argument("--ratio", type=float, default=0.25, help="fps sampling ratio")
    parser.add_argument("--num_SAModule", type=int, default=4, help="number of SAModule")
    parser.add_argument("--model_to_load", type=str, default=None, help="model for additional learning")
    parser.add_argument("--p2rank_col", type=str, default="probability", help="probability|score|zscore")
    parser.add_argument("--custom_net", type=str, default="False", help="custom PointNet or original PointNet")
    
    args = parser.parse_args()
    #batch_size = args.batch_size
    #num_workers = args.num_workers
    if args.valid_data == "None":
        args.valid_data = None
    #文字列をブール値に変換
    args.include_f1 = args.include_f1.lower() == 'true'
    args.custom_fps = args.custom_fps.lower() == 'true'
    args.rotInvLoss = args.rotInvLoss.lower() == 'true'
    args.pocket_segmented = args.pocket_segmented.lower() == 'true'
    args.channel_recalculate = args.channel_recalculate.lower() == 'true'
    args.p2rank_prediction_apply = args.p2rank_prediction_apply.lower() == 'true'
    args.p2rank_col = f" {args.p2rank_col}"
    args.custom_net = args.custom_net.lower() == 'true'
    
    
    utc_time = datetime.now(UTC)
    execution_time_str = (utc_time + timedelta(hours=9)).strftime('%Y%m%d_%H%M')
    if args.appendix != "":
        log_folder = f"./logs/{execution_time_str}_{args.appendix}"
        study_name = f"{execution_time_str}_{args.appendix}"
    else:
        log_folder = f"./logs/{execution_time_str}"
        study_name = f"{execution_time_str}"
    print(f"logs saved to {log_folder}")
    os.makedirs(log_folder,exist_ok=True)
    
    # 実行コマンドを保存
    log_command(f"{log_folder}/command.sh")
    
    # error and output to .log file
    # これによって出力やアウトプットがlogファイルに保存される
    out_f = open(f"{log_folder}/{sys.argv[0]}.log","w",encoding="utf-8")
    sys.stdout = out_f
    sys.stderr = out_f
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    logger.setLevel(logging.INFO)
    
    # 使用したファイルをlogに保存
    new_path = shutil.copy(f"{sys.argv[0]}",log_folder)
    logger.info(f"the file {sys.argv[0]} copied to {new_path}")
    parent_directory = '.'
    if '/' in sys.argv[0]:
        parent_directory = '/'.join(sys.argv[0].split('/')[:-1])
    
    # importの必要なモジュールも入れておく
    #os.makedirs(f"{log_folder}/dataset_util",exist_ok=True)
    dataset_util_path = shutil.copytree(f"{parent_directory}/dataset_util",f"{log_folder}/dataset_util")
    #os.makedirs(f"{log_folder}/models",exist_ok=True)
    models_path = shutil.copytree(f"{parent_directory}/models",f"{log_folder}/models")
    #study_name = 'distributed-study'
    python_exe_command_get_path = shutil.copy(f"{parent_directory}/python_exe_command_get.py", f"{log_folder}")
    
    # プロセス数（GPUの数と同じに設定）
    n_procs = 4
    logger.info(f"n_procs = {n_procs}")
    
    torch_fix_seed()
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

    # データとtrain test splitの取得
    if args.database_path is not None:
        pkl_data = torch.load(os.path.join(args.database_path,"database_to_use.pickle"))
        train_db, test_db = pkl_data["train_db"], pkl_data["test_db"]
        logger.debug(f"train_db:{train_db},test_db:{test_db}")
    elif args.valid_data is None:
        train_db, test_db = ToughM1(tough_data_dir=args.toughm1_data_root).get_structures_splits(fold_nr=args.fold_nr,moleculekit_version=args.moleculekit_version, include_f1=args.include_f1)
        logger.info("ToughM1().get_structures() is done")
        logger.debug(f"train_db:{train_db},test_db:{test_db}")
    else:
        train_db = ToughM1(tough_data_dir=args.toughm1_data_root).get_structures(valid_data=args.valid_data,moleculekit_version=args.moleculekit_version, include_f1=args.include_f1)
        test_db = [entry for entry in ToughM1(tough_data_dir=args.toughm1_data_root).get_structures(moleculekit_version=args.moleculekit_version, include_f1=args.include_f1) if entry not in train_db]
        logger.info("ToughM1().get_structures() is done")
        logger.debug(f"train_db:{train_db},test_db:{test_db}")
    # for debug デバッグするときは小さいデータで
    if args.debug_mode:
        train_db = random.sample(train_db,1000)
        test_db = random.sample(test_db,100)

    
    Objective(train_db,test_db)()

    out_f.close()
            
    
