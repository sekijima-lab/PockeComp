# PockeComp: Ultra-Fast Pocket Similarity Detection via Point Cloud Representation with Reinforced Ligand Probability

## How to use

### Build Environment

The Environment was built with conda.
```sh
conda create -n simsegpocket_pyg -y
conda activate simsegpocket_pyg

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
python -m pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
python -m pip install torch_geometric

conda install -c conda-forge biopython -y
conda install -c conda-forge rdkit -y
conda install -c conda-forge scikit-learn -y
conda install -c conda-forge biopandas -y
conda install -c conda-forge matplotlib -y

conda install -c conda-forge jupyterlab -y

conda install -c conda-forge openbabel -y
conda install -c conda-forge tqdm -y
```

### Train

```sh
# example command
conda activate simsegpocket
python train.py -a overall_v2_sigmoid_pos-normal_neg-relu_log_feat_overall_0_100_False_False_2_segmented_multiply_score_False_more_epoch --num_epoch 100 --cuda 0 --channel_recalculate False --fold_nr 0 --positive_loss normal --negative_loss relu_log --include_f1 True --last_act sigmoid --moleculekit_version v2 --custom_fps False --rotInvLoss False --p2rank_prediction_apply True --method segmented_multiply --p2rank_col score 
```

When the learning code is executed, an execution folder is generated under the ```./logs``` folder, in which a copy of the executed file and a CSV file containing the loss values ​​are saved.

### test

```sh
# example command
conda activate simsegpocket
python deeplytough_point_cloud_feat_customFPS_v2_eval.py --log_folder ./logs/... -c 0 -a model_100 --model_state_dict "model_100.pth
```
Select the execution folder path and the model to be verified and run the evaluation code.