#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh

conda remove -n simsegpocket_pyg --all -y
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
#conda install -c conda-forge ipympl -y

conda install -c conda-forge openbabel -y
conda install -c conda-forge tqdm -y

conda deactivate > /dev/null

