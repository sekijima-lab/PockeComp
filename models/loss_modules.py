import torch
from torch_scatter import scatter
import torch.nn.functional as F
from torch.nn import BCELoss

import logging
logger = logging.getLogger(__name__)

class DiceLoss(torch.nn.Module):
    def __init__(self,threshold_sum=False):
        super().__init__()
        self.threshold_sum = threshold_sum
        
    def forward(self, pred, target, batch, smooth=1):
        
        if self.threshold_sum:
            pred_flag = torch.where(pred > 0.5, 1., 1e-10)
            intersection = scatter(pred * target, batch, dim=0)
            dice = (2. * intersection + smooth) / (scatter(pred_flag, batch, dim=0) + scatter(target, batch, dim=0) + smooth)
        else:
            intersection = scatter(pred * target, batch, dim=0)
            dice = (2. * intersection + smooth) / (scatter(pred, batch, dim=0) + scatter(target, batch, dim=0) + smooth)
            
        #dice = (2.*intersection + smooth) / (pred.sum() + target.sum() + smooth)  
        
        return (1 - dice).sum()
 
class PocketCenterLoss(torch.nn.Module):
    def __init__(self, threshols_sum=False):
        super().__init__()
        self.threshold_sum = threshols_sum
        
    def forward(self,pred,target,batch,pos):
        logger.debug(f"batch.size()={batch.size()},pred.size()={pred.size()},target.size()={target.size()},pos.size()={pos.size()}")
        # Get unique batch indices to ensure consistent sizes
        unique_batches = torch.unique(batch)
        
        # Target center calculation
        sum_pos = scatter(pos[target==1], batch[target==1], dim=0, dim_size=len(unique_batches))
        to_divide = torch.bincount(batch[target==1], minlength=len(unique_batches))
        target_center = sum_pos / (to_divide[:,None] + 1e-10)  # Add small epsilon to avoid division by zero
        
        if self.threshold_sum:
            pred_flag = torch.where(pred > 0.5, 1., 1e-10)
            sum_pos = scatter(pos * pred_flag[:,None], batch, dim=0, dim_size=len(unique_batches))
            to_divide = scatter(pred_flag, batch, dim=0, dim_size=len(unique_batches))
            logger.debug(f"sum_pos:{sum_pos.size()}, to_divide:{to_divide.size()}")
            pred_center = sum_pos / (to_divide[:,None] + 1e-10)
        else:
            sum_pos = scatter(pos * pred[:,None], batch, dim=0, dim_size=len(unique_batches))
            to_divide = scatter(pred, batch, dim=0, dim_size=len(unique_batches))
            logger.debug(f"sum_pos:{sum_pos.size()}, to_divide:{to_divide.size()}")
            pred_center = sum_pos / (to_divide[:,None] + 1e-10)
            
        logger.debug(f"pred_center.size()={pred_center.size()}, target_center.size():{target_center.size()}")
        loss = torch.norm(target_center - pred_center)
        return loss
    
# VN-EGNN参考
class segmentation_loss_func(torch.nn.Module):
    def __init__(self,alpha=1,beta=1,cross_entropy=False,threshold_sum=False):
        super().__init__()
        self.dice_loss_func = DiceLoss(threshold_sum=threshold_sum)
        self.pocket_center_loss_func = PocketCenterLoss(threshols_sum=threshold_sum)
        self.cross_entropy = cross_entropy
        self.cross_entropy_loss_func = BCELoss()
        self.alpha = alpha
        self.beta = beta
        
    def forward(self,pred,target,batch,pos):
        pred = pred.view(-1).float()
        target = target.view(-1).float()
        logger.debug(f"pred={pred},target={target}")
        dice_loss = self.dice_loss_func(pred,target,batch)
        pocket_center_loss = self.pocket_center_loss_func(pred,target,batch,pos)
        #cross_entropy_loss = torch.tensor([self.cross_entropy_loss_func(pred[batch==i],target[batch==i]) for i in range(len(torch.unique(batch)))]).sum()
        if self.cross_entropy:
            cross_entropy_loss = self.cross_entropy_loss_func(pred,target)
        else:
            cross_entropy_loss = torch.tensor(0, requires_grad=False)
        return self.alpha * dice_loss+self.beta * pocket_center_loss+cross_entropy_loss, dice_loss, pocket_center_loss, cross_entropy_loss
    
    
class FeatureLossBase(torch.nn.Module):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
        
    def forward(self,x):
        return 0
    
class normal_positive_loss(FeatureLossBase):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return x
    
class relu_log_plus_positive_loss(FeatureLossBase):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return F.relu(torch.log(x/self.margin))
    
class relu_log_app_plus_positive_loss(FeatureLossBase):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return torch.log(1+x/self.margin)
    
class normal_negative_loss(torch.nn.Module):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return F.relu(self.margin - x)
    
class relu_log_minus_negative_loss(FeatureLossBase):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return F.relu(-torch.log(x/self.margin))
    
class relu_log_app_minus_negative_loss(FeatureLossBase):
    def __init__(self,margin=1):
        super().__init__()
        self.margin = margin
    def forward(self,x):
        return torch.log(1+self.margin/x)

class feature_loss_func(torch.nn.Module):
    def __init__(self,margin=1,positive_loss_type="normal",negative_loss_type="normal",reduction="sum",rotation_invariance_loss=False, rotation_invariance_loss_version=1):
        super().__init__()
        assert reduction=="sum" or reduction=="mean"
        self.reduction = reduction
        self.rotation_invariance_loss = rotation_invariance_loss
        self.rotation_invariance_loss_version = rotation_invariance_loss_version
        
        match positive_loss_type:
            case "normal":
                self.positive_loss = normal_positive_loss(margin)
            case "relu_log":
                self.positive_loss = relu_log_plus_positive_loss(margin)
            case "relu_log_app":
                self.positive_loss = relu_log_app_plus_positive_loss(margin)
        
        match negative_loss_type:
            case "normal":
                self.negative_loss = normal_negative_loss(margin)
            case "relu_log":
                self.negative_loss = relu_log_minus_negative_loss(margin)
            case "relu_log_app":
                self.negative_loss = relu_log_app_minus_negative_loss(margin)
                
    def forward(self,x_out,label):
        if not self.rotation_invariance_loss:
            if type(x_out) == tuple:
                x_out, _ = x_out
            x_out1, x_out2 = x_out[0::2], x_out[1::2]
            euclidean_diff = F.pairwise_distance(x_out1,x_out2)
            if self.reduction == "mean":
                return euclidean_diff, ((label * self.positive_loss(euclidean_diff)).pow(2) + ((1-label) * self.negative_loss(euclidean_diff)).pow(2)).mean()
            else:
                return euclidean_diff, ((label * self.positive_loss(euclidean_diff)).pow(2) + ((1-label) * self.negative_loss(euclidean_diff)).pow(2)).sum()
        else:
            if type(x_out) == tuple:
                x_out1, x_out2 = x_out
            else:
                raise ValueError(f"{x_out} is not tuple but feature_loss_func includes rotation invariance loss")
            
            x_out11, x_out12 = x_out1[0::2], x_out1[1::2]
            x_out21, x_out22 = x_out2[0::2], x_out2[1::2]
            
            if self.rotation_invariance_loss_version == 1: # hard
                euclidean_diff = F.pairwise_distance(x_out11,x_out12) + F.pairwise_distance(x_out21,x_out22)
            elif self.rotation_invariance_loss_version == 2: # soft, same as deeplytough
                euclidean_diff = F.pairwise_distance(x_out11,x_out12)
                
            rotation_invariance_loss_value = F.pairwise_distance(x_out11,x_out21) + F.pairwise_distance(x_out12,x_out22)
    
            if self.reduction == "mean":
                return euclidean_diff, rotation_invariance_loss_value.mean(), ((label * self.positive_loss(euclidean_diff)).pow(2) + ((1-label) * self.negative_loss(euclidean_diff)).pow(2) + rotation_invariance_loss_value).mean()
            else:
                return euclidean_diff, rotation_invariance_loss_value.sum(), ((label * self.positive_loss(euclidean_diff)).pow(2) + ((1-label) * self.negative_loss(euclidean_diff)).pow(2) + rotation_invariance_loss_value).sum()