#!/bin/bash

#SBATCH --mem=4GB                         # specify the needed memory
#SBATCH -p ml                             # specify partition
#SBATCH --gres=gpu:1                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                         # request 1 node
#SBATCH --time=11:00:00                   # runtime
#SBATCH -c 4                              # how many cores per task allocated
#SBATCH --mail-user=christoph.wagner@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_radarspeech
#SBATCH -o hpc_optim.out       # save output message under HLR_${SLURMJOBID}.out
#SBATCH -e hpc_optim.err       # save error messages under HLR_${SLURMJOBID}.err

module load modenv/ml
module load PyTorch scikit-learn matplotlib

python train_vocoder_params.py
  
exit 0