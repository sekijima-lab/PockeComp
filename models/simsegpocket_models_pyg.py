import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, PointNetConv, fps, global_max_pool, radius, knn, norm
from torch_geometric.typing import WITH_TORCH_CLUSTER
from torch_geometric.utils import scatter
from torch_scatter import scatter_max, scatter_mean
try:
    from .Custom_PointNetConv import _PointNetConv
except:
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Custom_PointNetConv import _PointNetConv
if not WITH_TORCH_CLUSTER:
    quit("This example requires 'torch-cluster'")

import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

def get_activation(activation_name):
    activations = {
        'relu': nn.ReLU(),
        'sigmoid': nn.Sigmoid(),
        'tanh': nn.Tanh(),
        'leaky_relu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'selu': nn.SELU(),
        'softmax': nn.Softmax(dim=-1),
        'none': nn.Identity()
    }
    return activations.get(activation_name.lower(), nn.Identity())

# 使用例
activation = get_activation('relu')
# または
activation = get_activation('sigmoid')

def optimized_fps_with_torch_geometric(pos, batch, ratio):
    # 点群の中心を計算
    center = scatter_mean(pos, batch, dim=0)
    logger.debug(f"center={center.size()}")
    # 各点と中心との距離を計算
    distances = torch.norm(pos - center[batch], dim=-1)
    
    # 中心から最も遠い点のインデックスを取得
    _, farthest_indices = scatter_max(distances, batch)
    
    # 最も遠い点を最初の点にするために、点群の順序を変更
    bincount = torch.bincount(batch)
    cumsum = torch.cumsum(bincount,dim=0) - bincount
    arange_all = torch.arange(pos.size(0), device=pos.device)
    
    # farthest_indicesを先頭に持ってくる
    new_order = torch.zeros_like(arange_all)
    new_order[cumsum] = farthest_indices
    
    # 他のindicesを整理
    mask1 = (arange_all <= farthest_indices[batch]) & (arange_all > cumsum[batch])
    mask2 = (arange_all > farthest_indices[batch])
    new_order[mask1] = arange_all[mask1] - 1
    new_order[mask2] = arange_all[mask2]
    logger.debug(f"new_order={new_order.size()}")
    reordered_pos = pos[new_order]
    reordered_batch = batch[new_order]
    
    # FPSを実行 (random_start=False)
    num_samples = int(pos.size(0) * ratio)
    sampled = fps(reordered_pos, reordered_batch, ratio=ratio, random_start=False)
    
    # 元の順序に戻す
    original_sampled = new_order[sampled]
    logger.debug(f"original_sampled per batch: {torch.bincount(batch[original_sampled])}")
    
    return original_sampled

# nearest neighbor でまとめる
class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn, custom_fps=False, custom_net=False):
        super().__init__()
        self.ratio = ratio
        self.r = r
        if custom_net:
            self.conv = _PointNetConv(nn, add_self_loops=False)
        else:
            self.conv = PointNetConv(nn, add_self_loops=False)
        self.custom_fps=custom_fps

    def forward(self, x, pos, batch):
        if self.custom_fps:
            idx = optimized_fps_with_torch_geometric(pos, batch, ratio=self.ratio)
        else:
            idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=32)
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = F.relu(self.conv((x, x_dst), (pos, pos[idx]), edge_index))
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch
    
# acidごとでまとめる(実装途中　6/6時点)
class AcidSAModule(torch.nn.Module):
    def __init__(self, ratio, nn):
        super().__init__()
        self.ratio = ratio
        self.conv = PointNetConv(nn, add_self_loops=False)

    def forward(self, x, pos, residue_number, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        #row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
        #                  max_num_neighbors=64)
        col = torch.arange(len())
        row = residue_number
        edge_index = torch.stack([col, row], dim=0)
        x_dst = None if x is None else x[idx]
        x = self.conv((x, x_dst), (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch

# 全部まとめる
class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn, dropout=0.5, custom_net=False):
        super().__init__()
        self.nn = nn
        self.custom_net = custom_net

    def forward(self, x, pos, batch):
        if self.custom_net:
            pos_j = pos
            pos_i = scatter_mean(pos,batch,dim=0)[batch]
            msg_norm = torch.linalg.norm(pos_j - pos_i,dim=1).unsqueeze(1)
            #print(f"msg_norm={msg_norm}")
            #logger.debug(f"edge_index_i={edge_index_i}, edge_index_j={edge_index_j}")
            knn_edges = knn(pos_j-pos_i, pos_j-pos_i, 8, batch, batch)
            logger.debug(f"knn_edges={knn_edges}")
            row, col = knn_edges
            v1 = pos_j[row] - pos_i[row]
            v2 = pos_j[col] - pos_i[col]
            cross = torch.cross(v1, v2, dim=1)  # 外積
            norm_cross = cross.norm(p=2, dim=1)  # 外積のノルム
            dot = (v1 * v2).sum(dim=1) # 内積
            norm_cross_ = scatter_mean(norm_cross, row).unsqueeze(1)
            dot_ = scatter_mean(dot, row).unsqueeze(1)
            logger.debug(msg_norm.size(),norm_cross_.size(),dot_.size(),x.size())
            msg = torch.cat([msg_norm, norm_cross_, dot_], dim=1)
            x = F.relu(self.nn(torch.cat([x, msg], dim=1)))
        else:
            x = F.relu(self.nn(torch.cat([x, pos], dim=1)))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


class feature_model(torch.nn.Module):
    def __init__(self, num_classes,in_channels,num_SAModule=2,dropout=0.5):
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, in_channels))
        self.register_buffer('scaler_std', torch.ones(1, in_channels))
        self.in_channels = in_channels
        # Input channels account for both `pos` and node features.
        self.sa_modules = torch.nn.ModuleList([])
        ratio = 0.25
        radius = 1.0
        hidden = 16
        for i in range(num_SAModule):
            # MLPはデフォルトで最後ReLU関数をかけている
            self.sa_modules.append(SAModule(ratio, radius, MLP([in_channels + 3, hidden, hidden, 2*hidden],act='sigmoid')))
            #ratio /= 2
            radius *= 2
            hidden *= 2
            in_channels = hidden
        self.global_sa_module = GlobalSAModule(MLP([in_channels + 3, hidden, hidden, 2*hidden],act='sigmoid'))
        hidden *= 2

        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, pos, batch):
        x = (x - self.scaler_mean.expand(x.size()[0],self.in_channels)) / self.scaler_std.expand(x.size()[0],self.in_channels)
        logger.debug(f"batch={batch}")
        sa_out = (x, pos, batch)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"batch={batch}")
        sa_out = self.global_sa_module(*sa_out)
        x, pos, batch = sa_out
        logger.debug(f"batch={batch}")

        return F.softmax(x)
    


class feature_model_v2(torch.nn.Module):
    def __init__(self, out_channels,in_channels,num_SAModule=2,dropout=0.5,last_act="relu", custom_fps=False, ratio=0.25, custom_net=False, **kwargs):
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, in_channels))
        self.register_buffer('scaler_std', torch.ones(1, in_channels))
        self.in_channels = in_channels
        # Input channels account for both `pos` and node features.
        self.sa_modules = torch.nn.ModuleList([])
        ratio = ratio
        radius = 2.0
        hidden = 64
        for i in range(num_SAModule):
            # MLPはデフォルトで最後ReLU関数をかけている
            self.sa_modules.append(SAModule(ratio, radius, 
                                        torch.nn.Sequential(torch.nn.Linear(in_channels+3,hidden),
                                                            torch.nn.Sigmoid(),
                                                            torch.nn.Linear(hidden, 2*hidden)),
                                        custom_fps, custom_net=custom_net))
            #ratio /= 2
            radius *= 2
            hidden *= 2
            in_channels = hidden
        self.global_sa_module = GlobalSAModule(torch.nn.Sequential(torch.nn.Linear(in_channels+3,hidden),
                                                               torch.nn.Sigmoid(),
                                                               torch.nn.Linear(hidden, 2*hidden)),custom_net=custom_net)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden,hidden),
            torch.nn.Sigmoid(),
            torch.nn.Linear(hidden,out_channels)
        )
            
        self.last_act = get_activation(last_act)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, pos, batch):
        logger.debug(f"Input tensor shapes before scaling:")
        logger.debug(f"x: {x.size()}, scaler_mean: {self.scaler_mean.size()}, scaler_std: {self.scaler_std.size()}")
        logger.debug(f"in_channels: {self.in_channels}")
        logger.debug(f"x first few values: {x[:5]}")
        logger.debug(f"scaler_mean: {self.scaler_mean}")
        logger.debug(f"scaler_std: {self.scaler_std}")
        x = (x - self.scaler_mean.expand(x.size()[0],self.in_channels)) / self.scaler_std.expand(x.size()[0],self.in_channels)
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = (x, pos, batch)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = self.global_sa_module(*sa_out)
        x, pos, batch = sa_out
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        x = self.fc(x)
        x = self.last_act(x)
        return x

class feature_model_v3(torch.nn.Module):
    def __init__(self, out_channels,in_channels,num_SAModule=2,dropout=0.5,last_act="relu", custom_fps=False, **kwargs):
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, in_channels))
        self.register_buffer('scaler_std', torch.ones(1, in_channels))
        self.in_channels = in_channels
        # Input channels account for both `pos` and node features.
        self.sa_modules = torch.nn.ModuleList([])
        ratio = 0.25
        radius = 2.0
        hidden = 64
        for i in range(num_SAModule):
            # MLPはデフォルトで最後ReLU関数をかけている
            self.sa_modules.append(SAModule(ratio, radius, 
                                           torch.nn.Sequential(torch.nn.Linear(in_channels+3,hidden),
                                                               torch.nn.Sigmoid(),
                                                               torch.nn.Linear(hidden, 2*hidden)),
                                           custom_fps))
            #ratio /= 2
            radius *= 2
            hidden *= 2
            in_channels = hidden
        self.global_sa_module = GlobalSAModule(torch.nn.Sequential(torch.nn.Linear(in_channels+3,hidden),
                                                               torch.nn.Sigmoid(),
                                                               torch.nn.Linear(hidden, 2*hidden)))
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden,hidden),
            torch.nn.Linear(hidden,out_channels)
        )
            
        self.last_act = get_activation(last_act)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, pos, batch):
        logger.debug(f"Input tensor shapes before scaling:")
        logger.debug(f"x: {x.size()}, scaler_mean: {self.scaler_mean.size()}, scaler_std: {self.scaler_std.size()}")
        logger.debug(f"in_channels: {self.in_channels}")
        logger.debug(f"x first few values: {x[:5]}")
        logger.debug(f"scaler_mean: {self.scaler_mean}")
        logger.debug(f"scaler_std: {self.scaler_std}")
        x = (x - self.scaler_mean.expand(x.size()[0],self.in_channels)) / self.scaler_std.expand(x.size()[0],self.in_channels)
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = (x, pos, batch)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = self.global_sa_module(*sa_out)
        x, pos, batch = sa_out
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        x = self.fc(x)
        x = self.last_act(x)
        return x


class feature_model_v4(torch.nn.Module):
    def __init__(self, out_channels,in_channels,num_SAModule=2,dropout_first=0.0,dropout=0.0,last_act="sigmoid",act="relu",custom_fps=False, **kwargs):
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, in_channels))
        self.register_buffer('scaler_std', torch.ones(1, in_channels))
        self.in_channels = in_channels
        # Input channels account for both `pos` and node features.
        self.sa_modules = torch.nn.ModuleList([])
        ratio = 0.25
        radius = 2.0
        hidden = 64
        for i in range(num_SAModule):
            # MLPはデフォルトで最後ReLU関数をかけている

            if i == 0:
                self.sa_modules.append(SAModule(ratio, radius, 
                                        MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act, dropout=dropout_first),
                                        custom_fps=custom_fps))
            else:
                self.sa_modules.append(SAModule(ratio, radius, 
                                        MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act, dropout=dropout),
                                        custom_fps=custom_fps))
            #ratio /= 2
            radius *= 2
            hidden *= 2
            in_channels = hidden
        self.global_sa_module = GlobalSAModule(MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act, dropout=dropout))
        
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(2*hidden,hidden),
            torch.nn.Linear(hidden,out_channels)
        )
            
        self.last_act = get_activation(last_act)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, pos, batch):
        x = (x - self.scaler_mean.expand(x.size()[0],self.in_channels)) / self.scaler_std.expand(x.size()[0],self.in_channels)
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = (x, pos, batch)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = self.global_sa_module(*sa_out)
        x, pos, batch = sa_out
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        x = self.fc(x)
        x = self.last_act(x)
        return x
    
# 残基プーリング
class feature_model_v5(torch.nn.Module):
    def __init__(self, out_channels,in_channels,num_SAModule=2,dropout_first=0.0,dropout=0.0,last_act="relu",act="sigmoid",custom_fps=False):
        super().__init__()
        self.register_buffer('scaler_mean', torch.zeros(1, in_channels))
        self.register_buffer('scaler_std', torch.ones(1, in_channels))
        self.in_channels = in_channels
        # Input channels account for both `pos` and node features.
        self.sa_modules = torch.nn.ModuleList([])
        ratio = 0.5
        hidden = 16
        self.acid_sa_module = AcidSAModule(MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act, dropout=dropout_first))
        hidden *= 2
        in_channels = hidden
        radius = 8.0
        for i in range(num_SAModule):
            # MLPはデフォルトで最後ReLU関数をかけている
            self.sa_modules.append(SAModule(ratio, radius, MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act, dropout=dropout),custom_fps=custom_fps))
            #ratio /= 2
            radius *= 2
            hidden *= 2
            in_channels = hidden
        self.global_sa_module = GlobalSAModule(MLP([in_channels + 3, hidden, hidden, hidden * 2], act=act))
        self.fc = MLP([hidden * 2, hidden, hidden, out_channels],act=None)
        self.last_act = get_activation(last_act)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, x, pos, batch, res_id):
        x = (x - self.scaler_mean.expand(x.size()[0],self.in_channels)) / self.scaler_std.expand(x.size()[0],self.in_channels)
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        x, pos, batch, res_id = self.acid_sa_module(x, pos, batch, res_id)
        sa_out = (x, pos, batch)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        sa_out = self.global_sa_module(*sa_out)
        x, pos, batch = sa_out
        logger.debug(f"x={x.size()},pos={pos.size()},batch={batch.size()}")
        x = self.fc(x)
        x = self.last_act(x)
        return x 

    
class segmentation_model(torch.nn.Module):
    def __init__(self,in_channels,num_SAModule=2,dropout_first=0.0, dropout=0.5, act='sigmoid', custom_fps=False, **kwargs):
        super().__init__()
        self.sa_modules = torch.nn.ModuleList()
        ratio = 0.25
        radius = 1.0
        hidden = 16
        in_channels_list = [in_channels]
        for i in range(num_SAModule):
            if i==0:
                self.sa_modules.append(SAModule(ratio, radius, MLP([in_channels + 3, hidden, hidden, hidden * 2], 
                                            act=act,dropout=dropout_first,add_self_loops=False), custom_fps=custom_fps))
            else:
                self.sa_modules.append(SAModule(ratio, radius, MLP([in_channels + 3, hidden, hidden, hidden * 2], 
                                            act=act,dropout=dropout,add_self_loops=False), custom_fps=custom_fps))
            radius *= 2
            hidden *= 2
            in_channels = hidden
            in_channels_list.insert(0,in_channels)
        #self.middle_layer = MLP([in_channels + 3, hidden, hidden, hidden * 2], add_self_loops=True)
        self.point_net_convs = torch.nn.ModuleList()
        self.up_sample_convs = torch.nn.ModuleList()
        for i in range(num_SAModule):
            self.point_net_convs.append(PointNetConv(local_nn=MLP([in_channels_list[i] +3,
                                                                   hidden, hidden, 
                                                                   in_channels_list[i+1]],act=act, dropout=dropout),add_self_loops=True))
            hidden = in_channels_list[i+1]
            self.up_sample_convs.append(torch.nn.Sequential(
                torch.nn.Linear(hidden*2,hidden),
                norm.BatchNorm(hidden),
                torch.nn.ReLU()
            ))
        self.fc = MLP([in_channels_list[-1],hidden,hidden,1],dropout=dropout)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self,x,pos,batch):
        logger.debug(f"x.size()={x.size()}")
        sa_out_list = []
        sa_out = (x, pos, batch)
        sa_out_list.append(sa_out)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x.size()={x.size()}")
            sa_out_list.append(sa_out)
        xl, posl, batchl = sa_out_list.pop()
        for point_net_conv, up_sample_conv in zip(self.point_net_convs, self.up_sample_convs):
            sa_out = sa_out_list.pop()
            x, pos, batch = sa_out
            row, col = knn(pos,posl,64,batch,batchl)
            edge_index = torch.stack([row,col], dim=0)
            x_new = point_net_conv((xl,x),(posl,pos),edge_index)
            x_concat = torch.cat([x_new,x], dim=1)
            logger.debug(f"x.size()={x.size()},xl.size()={xl.size()},x_new.size()={x_new.size()},x_concat.size()={x_concat.size()}")
            xl = up_sample_conv(x_concat)
            posl, batchl = pos, batch
        logger.debug(xl.size(),posl.size(),batch.size())
        xl = self.fc(xl,batchl)
        xl = F.sigmoid(xl)
        return xl, posl, batchl

class segmentation_model_v2(torch.nn.Module):
    def __init__(self,in_channels,num_SAModule=2,dropout=0.5):
        super().__init__()
        self.sa_modules = torch.nn.ModuleList()
        ratio = 0.25
        radius = 1.0
        hidden = 16
        in_channels_list = [in_channels]
        for i in range(num_SAModule):
            self.sa_modules.append(SAModule(ratio, radius, torch.nn.Sequential(torch.nn.Linear(in_channels+3,hidden),
                                                               torch.nn.Sigmoid(),
                                                               torch.nn.Linear(hidden, 2*hidden))))
            radius *= 2
            hidden *= 2
            in_channels = hidden
            in_channels_list.insert(0,in_channels)
        #self.middle_layer = MLP([in_channels + 3, hidden, hidden, hidden * 2], add_self_loops=True)
        self.point_net_convs = torch.nn.ModuleList()
        self.up_sample_convs = torch.nn.ModuleList()
        for i in range(num_SAModule):
            self.point_net_convs.append(PointNetConv(local_nn=MLP([in_channels_list[i] +3,
                                                                   hidden, hidden, 
                                                                   in_channels_list[i+1]],act='sigmoid'),add_self_loops=True))
            hidden = in_channels_list[i+1]
            self.up_sample_convs.append(torch.nn.Sequential(
                torch.nn.Linear(hidden*2,hidden),
                norm.BatchNorm(hidden),
                torch.nn.ReLU()
            ))
        self.fc = MLP([in_channels_list[-1],hidden,hidden,1],dropout=dropout)
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def forward(self,x,pos,batch):
        logger.debug(f"x.size()={x.size()}")
        sa_out_list = []
        sa_out = (x, pos, batch)
        sa_out_list.append(sa_out)
        for sa_module in self.sa_modules:
            sa_out = sa_module(*sa_out)
            x, pos, batch = sa_out
            logger.debug(f"x.size()={x.size()}")
            sa_out_list.append(sa_out)
        xl, posl, batchl = sa_out_list.pop()
        for point_net_conv, up_sample_conv in zip(self.point_net_convs, self.up_sample_convs):
            sa_out = sa_out_list.pop()
            x, pos, batch = sa_out
            row, col = knn(pos,posl,64,batch,batchl)
            edge_index = torch.stack([row,col], dim=0)
            x_new = point_net_conv((xl,x),(posl,pos),edge_index)
            x_concat = torch.cat([x_new,x], dim=1)
            logger.debug(f"x.size()={x.size()},xl.size()={xl.size()},x_new.size()={x_new.size()},x_concat.size()={x_concat.size()}")
            xl = up_sample_conv(x_concat)
            posl, batchl = pos, batch
        logger.debug(xl.size(),posl.size(),batch.size())
        xl = self.fc(xl,batchl)
        xl = F.sigmoid(xl)
        return xl, posl, batchl
