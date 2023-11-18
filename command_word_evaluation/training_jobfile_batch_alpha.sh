#!/bin/bash

#SBATCH --mem=6GB                        # specify the needed memory
#SBATCH -p alpha                          # specify partition
#SBATCH --gres=gpu:1                      # use 1 GPU per node (i.e. use one GPU per task)
#SBATCH --nodes=1                         # request 1 node
#SBATCH --time=5:00:00                    # runtime
#SBATCH -c 8                              # how many cores per task allocated
#SBATCH --mail-user=christoph.wagner@tu-dresden.de   # email address
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH -A p_radarspeech

module load Python
source .../speech_synthesis_env/bin/activate

module load modenv/hiera
module load GCC/10.2.0 CUDA/11.1.1 OpenMPI/4.0.5 PyTorch/1.9.0 tqdm/4.56.2

python train_vocoder_params_lstm.py
  
exit 0


