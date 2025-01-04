#!/bin/bash

#SBATCH --partition=rome
#SBATCH --ntasks-per-node=128
#SBATCH --nodes=1
#SBATCH --time=2:00:00

cd /home/i/ioannisam/PR-ML

module load gcc/12.2.0 python/3.10.10
source /home/i/ioannisam/myenv/bin/activate

python KNNTuning.py
python RFTuning.py
python NNTuning.py

deactivate