import importlib
import sys
import os
import argparse
import torch
import importlib.util as iu
import sys
from dataset_util.toughm1_dataset_for_pyg_8dim import TOUGH_M1_triplet_dataset, VERTEX_triplet_dataset, PROSPECCTS_triplet_dataset,TOUGH_M1_pair_dataset_eval, VERTEX_pair_dataset_eval, PROSPECCTS_pair_dataset_eval
from dataset_util.Vertex_utils import Vertex
from dataset_util.Prospeccts_utils import Prospeccts
from torch_geometric.loader import DataLoader
import logging
import pandas as pd
from datetime import datetime, UTC, timedelta
import shutil
from collections import defaultdict
from sklearn.metrics import roc_auc_score
import yaml
import traceback
import math
from tqdm import tqdm
import numpy as np

logger = logging.getLogger(__name__)

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
    

def center_to_origin(coords):
    center = torch.mean(coords, dim=0, keepdim=True)
    coords = coords - center
    return coords

def Rotation_xyz(coords):
    # 原点中心になるように平行移動
    coords = center_to_origin(coords)
    
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--log_folder",default=None,help="train log folder")
    parser.add_argument("--hyperparameters_yml", default="feature_hyperparameters.yml")
    parser.add_argument("--data_file",default="database_to_use.pickle")
    parser.add_argument("--model_file",default="./models/simsegpocket_models_pyg.py")
    parser.add_argument("--model_name",default="feature_model_v2",help="trained model")
    parser.add_argument("--model_state_dict",default=None,help="saved state dict")
    parser.add_argument('-a', '--appendix', default="", help="フォルダ名につけるappendix")
    parser.add_argument('-c', '--cuda', type=int, default=4, help="使用するCUDAデバイスの指定")
    parser.add_argument('-d', '--debug_mode', type=bool, default=False, help="デバッグモードでやるかどうか")
    parser.add_argument('--channel_recalculate', type=bool, default=False,help="DeeplyToughと同様にチャンネル値を計算し直すか")
    parser.add_argument('--toughm1_data_root', default="./TOUGH-M1/", help="TOUGH-M1データセットディレクトリの場所")
    parser.add_argument("--seed", default=0)
    parser.add_argument("--batch_size", default=7)
    parser.add_argument("--num_sa_modules", default=4)
    parser.add_argument("--dropout", default=0.25)
    parser.add_argument("--moleculekit_version", default="v2")
    parser.add_argument("--include_f1", default=True)
    parser.add_argument('--last_act', type=str, default="relu", help="feature_model_v2の最終層にreluをかけるか否か")
    parser.add_argument("--custom_fps_force", type=str, default="False")
    
    args = parser.parse_args()
    
    args.custom_fps_force = args.custom_fps_force.lower() == "true"
    
    utc_time = datetime.now(UTC)
    execution_time_str = (utc_time + timedelta(hours=9)).strftime('%Y%m%d_%H%M')
    if args.appendix != "":
        log_folder = f"{args.log_folder}/evaluate{execution_time_str}_{args.appendix}"
    else:
        log_folder = f"{args.log_folder}/evaluate{execution_time_str}"
    print(f"logs saved to {log_folder}")
    os.makedirs(log_folder,exist_ok=True)
    
    # error and output to .log file
    # これによって出力やアウトプットがlogファイルに保存される
    out_f = open(f"{log_folder}/{sys.argv[0]}.log","w",encoding="utf-8")
    sys.stdout = out_f
    sys.stderr = out_f
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if args.debug_mode:
        logging.basicConfig(level=logging.DEBUG, format=log_format)
    else:
        logging.basicConfig(level=logging.INFO, format=log_format)
    #logger.setLevel(logging.INFO)
    
    # 使用したファイルをlogに保存
    new_path = shutil.copy(f"{sys.argv[0]}",log_folder)
    logger.info(f"the file {sys.argv[0]} copied to {new_path}")
    parent_directory = '.'
    if '/' in sys.argv[0]:
        parent_directory = '/'.join(sys.argv[0].split('/')[:-1])
    dataset_util_path = shutil.copytree(f"{parent_directory}/dataset_util",f"{log_folder}/dataset_util")
    
    ### ここでhyperparameter.ymlから情報取得 ###
    data_file = args.data_file
    model_file = args.model_file
    model_name = args.model_name
    if args.hyperparameters_yml:
        with open(f'{args.log_folder}/{args.hyperparameters_yml}', 'r') as file:
            hyperparameters = yaml.safe_load(file)
    
        # 情報取得
        data_file = hyperparameters["logs"]["data_file"]
        model_file = hyperparameters["logs"]["model_file"]
        model_name = hyperparameters["logs"]["model_name"]
        model_state_dict = hyperparameters["logs"]["model_state_dict"]
        
        num_SAModule = hyperparameters["model_feature_parameters"]["num_SAModule"]
        dropout = hyperparameters["model_feature_parameters"]["dropout"]
        in_channels = hyperparameters["model_feature_parameters"]["in_channels"]
        out_channels = hyperparameters["model_feature_parameters"]["out_channels"]
        last_act = hyperparameters["model_feature_parameters"]["last_act"]
        custom_fps = hyperparameters["model_feature_parameters"]["custom_fps"]
        ratio = hyperparameters["model_feature_parameters"].get("ratio",0.25)
        custom_net = hyperparameters["model_feature_parameters"].get("custom_net",False)
        
        data_root = hyperparameters["dataset_settings"]["data_root"]
        recalculate = hyperparameters["dataset_settings"]["recalculate"]
        valid_data = hyperparameters["dataset_settings"]["valid_data"]
        moleculekit_version = hyperparameters["dataset_settings"]["moleculekit_version"]
        include_f1 = hyperparameters["dataset_settings"]["include_f1"]
        if isinstance(include_f1, str):
            include_f1 = include_f1 == "True"
        pocket_segmented = hyperparameters["dataset_settings"]["pocket_segmented"]
        p2rank_col = hyperparameters["dataset_settings"].get("p2rank_col",' probability')
        
        batch_size = hyperparameters["dataloader_settings"]["batch_size"]
        num_workers = hyperparameters["dataloader_settings"]["num_workers"]
        
        rotation_invariance_loss = hyperparameters["lossfunc_settings"].get("rotation_invariance_loss",False)
        rotation_invariance_loss_version = hyperparameters["lossfunc_settings"].get("rotation_invariance_loss_version",1)
        
        p2rank_prediction_apply = hyperparameters["dataset_settings"].get("p2rank_prediction_apply", False)
        
        method = hyperparameters["other_settings"].get("method", "segmented")
        if method=="segmented":
            pocket_segmented = True
        else:
            pocket_segmented = False
            
        
    if args.model_state_dict is not None:
        model_state_dict = args.model_state_dict
    
    
    
    feature_model_v2 = get_model_object(args.log_folder,model_file,model_name)
    
    data = torch.load(os.path.join(args.log_folder,data_file))
    
    device = f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu"
    model_feature = None
    def eval_dataset_and_save(eval_dataset, key, pair_dataset=None):
        torch.set_num_threads(8)
        
        with torch.no_grad():
            eval_dataset_effective_list = eval_dataset.get_effective_pdb_list()
            eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
            feature_dict = {}
            # 特徴量をあらかじめ計算
            logger.info("Calculate Fature First")
            for data in tqdm(eval_data_loader):
                x, pos, batch = data.x.to(device), data.pos.to(device), data.batch.to(device)
                if "rot" in key:
                    pos = Rotation_xyz(data.pos)
                    pos = pos.to(device)
                if method=="segmented_add":
                    atom_feature = torch.cat([data.x, data.pocket_flag.unsqueeze(1)], dim=1)
                    atom_feature = atom_feature.to(device)
                elif method=="segmented_multiply":
                    atom_feature = data.x * data.pocket_flag.unsqueeze(1)
                    atom_feature = atom_feature.to(device)
                else:    
                    atom_feature = data.x.to(device)
                x = atom_feature
                x_feature = model_feature(x,pos,batch)
                x_feature = x_feature.detach().cpu()
                for i,code in enumerate(data.code5):
                    feature_dict[code] = x_feature[i,:]
            
            all_features = torch.stack([feature_dict[code] for code in eval_dataset_effective_list])
            torch.save({
                "code_list":eval_dataset_effective_list,
                "feature_list":all_features,
                "feature_dict":feature_dict
            },f"{log_folder}/{key}_code_feature.pickle")
            distances = torch.cdist(all_features, all_features)
            # 各要素ごとに計算
            positive_score_list = [] #小さくなった方がいい
            negative_score_list = [] #大きくなった方がいい
            codes = []
            
            
            logger.info("Scoring pocket dist")
            pocket1_list = []
            pocket2_list = []
            dist_list = []
            check_dict = defaultdict(lambda:False)
            rank_list = []
            flags = []
            for i in tqdm(range(len(eval_dataset))):
                data = eval_dataset.get_triplet_info(i)
                code = data["code"]
                positive_idx = data["positive_list"].tolist()
                negative_idx = data["negative_list"].tolist()
                rank_per_idx = torch.argsort(distances)
                
                try:
                    positive_score = (rank_per_idx[i,positive_idx] / len(eval_dataset)).mean()
                    positive_score_list.append(positive_score.item())
                except:
                    positive_score_list.append(-1)
                try:
                    negative_score = 1 - (rank_per_idx[i,negative_idx] / len(eval_dataset)).mean()
                    negative_score_list.append(negative_score.item())
                except:
                    negative_score_list.append(-1)
                codes.append(data["code"])
                
                for j in positive_idx:
                    if check_dict[(j,i)]:
                        continue
                    pocket1_list.append(code)
                    pocket2_list.append(eval_dataset_effective_list[j])
                    dist_list.append(distances[i,j].item())
                    rank_list.append(rank_per_idx[i,j].item())
                    flags.append(1)
                    check_dict[(i,j)] = True
                    
                for j in negative_idx:
                    if check_dict[(j,i)]:
                        continue
                    pocket1_list.append(code)
                    pocket2_list.append(eval_dataset_effective_list[j])
                    dist_list.append(distances[i,j].item())
                    rank_list.append(rank_per_idx[i,j].item())
                    flags.append(0)
                    check_dict[(i,j)] = True
                    
            eval_result = pd.DataFrame({"code":codes,"positive_score":positive_score_list,"negative_score":negative_score_list})
            eval_result.to_csv(f"{log_folder}/{key}_score_result_{model_name}_{model_state_dict}.csv",encoding='utf-8')    
            dist_data = pd.DataFrame({"pocket1":pocket1_list,"pocket2":pocket2_list,"label":flags,"dist":dist_list,"rank":rank_list})
            dist_data.to_csv(f"{log_folder}/{key}_dist_result_{model_name}_{model_state_dict}.csv",encoding='utf-8')    
                    
            y_true = dist_data["label"]
            y_pred_proba = -dist_data["dist"]
            
            logger.info(f"AUC Score of {key} (Not Random) : {roc_auc_score(y_true,y_pred_proba)}")        
                
        with torch.no_grad():
            if pair_dataset is not None:
                pocket1_list = []
                pocket2_list = []
                dist_list = []
                flags = []
                eval_dataset = pair_dataset
                eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
                for data in eval_data_loader:
                    x, pos, batch = data.x.to(device), data.pos.to(device), data.batch_sub.to(device)
                    if method=="segmented_add":
                        atom_feature = torch.cat([data.x, data.pocket_flag.unsqueeze(1)], dim=1)
                        atom_feature = atom_feature.to(device)
                    elif method=="segmented_multiply":
                        atom_feature = data.x * data.pocket_flag.unsqueeze(1)
                        atom_feature = atom_feature.to(device)
                    else:    
                        atom_feature = data.x.to(device)
                    x = atom_feature
                    x_feature = model_feature(x,pos,batch)
                    x_feature = x_feature.detach().cpu()
                    flags.extend(data.label.tolist())
                    pocket1_list.extend([eval_dataset._idx_map[code] for code in data.code5[0::2].tolist()])
                    pocket2_list.extend([eval_dataset._idx_map[code] for code in data.code5[1::2].tolist()])
                    dist_list.extend(torch.linalg.norm(x_feature[0::2,:]-x_feature[1::2,:],dim=1).tolist())
                    
                dist_data = pd.DataFrame({"pocket1":pocket1_list,"pocket2":pocket2_list,"label":flags,"dist":dist_list})
                dist_data.to_csv(f"{log_folder}/{key}_dist_result_{model_name}_{model_state_dict}_random.csv",encoding='utf-8') 
                

                y_true = dist_data["label"]
                y_pred_proba = -dist_data["dist"]
                
                logger.info(f"AUC Score of {key} (Random) : {roc_auc_score(y_true,y_pred_proba)}")

    if args.custom_fps_force:
        custom_fps = True
    model_feature = feature_model_v2(out_channels,in_channels,num_SAModule=num_SAModule,dropout=dropout,last_act=last_act,custom_fps=custom_fps,ratio=ratio,custom_net=custom_net)
    model_feature.load_state_dict(torch.load(os.path.join(args.log_folder,model_state_dict)))
    model_feature = model_feature.to(device)
    model_feature.eval()
    if valid_data is None:
        try:
            fold_nr = hyperparameters["dataset_settings"]["fold_nr"]
            test_data = torch.load(args.toughm1_data_root.replace("/TOUGH-M1/",f"/test_dataset_{fold_nr}.pt"))
            assert test_data["data_root"] == args.toughm1_data_root
            eval_dataset = test_data["triplet_dataset"]
            eval_dataset.set_col(p2rank_col)
            eval_dataset.set_recalculate(recalculate)
            #pair_dataset = test_data["dataset"]
            #pair_dataset.set_col(p2rank_col)
            #pair_dataset.set_recalculate(recalculate)
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1_sub")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1_rot")
        except:
            logger.info(f"{traceback.format_exc()}, eval_dataset construct")
            test_db = data["test_db"]
            logger.debug(f"test_db={test_db}")
            eval_dataset = TOUGH_M1_triplet_dataset(test_db, data_root=data_root, recalculate=False, col=p2rank_col)
            pair_dataset = TOUGH_M1_pair_dataset_eval(test_db, data_root=data_root, recalculate=False, col=p2rank_col)
            torch.save({
                "dataset":pair_dataset, "triplet_dataset":eval_dataset,"data_root":data_root, "recalculate":False
            }, args.toughm1_data_root.replace("/TOUGH-M1/",f"/test_dataset_{fold_nr}.pt"))
            eval_dataset.set_recalculate(recalculate)
            #pair_dataset.set_recalculate(recalculate)
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1_sub")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="TOUGH-M1_rot")
    elif valid_data=="vertex":
        try:
            test_data = torch.load(args.toughm1_data_root.replace("/TOUGH-M1/","/vertex_dataset.pt"))
            assert test_data["recalculate"] == recalculate and test_data["data_root"] == args.toughm1_data_root
            eval_dataset = test_data["triplet_dataset"]
            eval_dataset.set_col(p2rank_col)
            eval_dataset.set_recalculate(recalculate)
            #pair_dataset = test_data["dataset"]
            #pair_dataset.set_col(p2rank_col)
            #pair_dataset.set_recalculate(recalculate)
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex_sub")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex_rot")
        except:
            logger.info(f"{traceback.format_exc()}, eval_dataset construct")
            test_db = Vertex(vertex_data_dir=args.toughm1_data_root.replace("/TOUGH-M1/","/Vertex/")).get_structures(extra_mappings=True, moleculekit_version=moleculekit_version, include_f1=include_f1)
            logger.debug(f"test_db={test_db}")
            eval_dataset = VERTEX_triplet_dataset(test_db, data_root=args.toughm1_data_root.replace("/TOUGH-M1/","/Vertex/"), recalculate=False, col=p2rank_col)
            pair_dataset = VERTEX_pair_dataset_eval(test_db, data_root=args.toughm1_data_root.replace("/TOUGH-M1/","/Vertex/"), recalculate=False, col=p2rank_col)
            torch.save({
                "dataset":pair_dataset, "triplet_dataset":eval_dataset,"data_root":data_root, "recalculate":False
            },args.toughm1_data_root.replace("/TOUGH-M1/","/vertex_dataset.pt"))
            eval_dataset.set_recalculate(recalculate)
            #pair_dataset.set_recalculate(recalculate)
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex_sub")
            eval_dataset_and_save(eval_dataset=eval_dataset, key="vertex_rot")
    elif valid_data=="prospeccts":
        
        for dbname in Prospeccts.dbnames:
            try:
                test_data = torch.load(args.toughm1_data_root.replace("/TOUGH-M1/",f"/prospeccts_dataset_{dbname}.pt"))
                assert test_data["recalculate"] == recalculate and test_data["data_root"] == args.toughm1_data_root
                eval_dataset = test_data["triplet_dataset"]
                eval_dataset.set_col(p2rank_col)
                eval_dataset.set_recalculate(recalculate)
                #pair_dataset = test_data["dataset"]
                #pair_dataset.set_col(p2rank_col)
                #pair_dataset.set_recalculate(recalculate)
                eval_dataset_and_save(eval_dataset=eval_dataset,key=dbname)
                eval_dataset_and_save(eval_dataset=eval_dataset, key=f"{dbname}_sub")
                eval_dataset_and_save(eval_dataset=eval_dataset,key=f"{dbname}_rot")
            except:
                logger.info(f"{traceback.format_exc()}, eval_dataset construct")
                test_db = Prospeccts(prospeccts_data_dir=args.toughm1_data_root.replace("/TOUGH-M1/","/prospeccts/"),dbname=dbname).get_structures(moleculekit_version=moleculekit_version, include_f1=include_f1)
                logger.debug(f"test_db={test_db}")
                eval_dataset = PROSPECCTS_triplet_dataset(test_db, dbname=dbname, data_root=args.toughm1_data_root.replace("/TOUGH-M1/","/prospeccts/"),recalculate=False, col=p2rank_col)
                pair_dataset = PROSPECCTS_pair_dataset_eval(test_db, dbname=dbname, data_root=args.toughm1_data_root.replace("/TOUGH-M1/","/prospeccts/"),recalculate=False, col=p2rank_col)
                torch.save({
                    "dataset":pair_dataset, "triplet_dataset":eval_dataset,"data_root":data_root, "recalculate":recalculate
                }, args.toughm1_data_root.replace("/TOUGH-M1/",f"/prospeccts_dataset_{dbname}.pt"))
                eval_dataset.set_recalculate(recalculate)
                #pair_dataset.set_recalculate(recalculate)
                eval_dataset_and_save(eval_dataset=eval_dataset,key=dbname)
                eval_dataset_and_save(eval_dataset=eval_dataset, key=f"{dbname}_sub")
                eval_dataset_and_save(eval_dataset=eval_dataset,key=f"{dbname}_rot")
    torch.cuda.empty_cache()
    out_f.close()
    

        
    
        
        
        
            
