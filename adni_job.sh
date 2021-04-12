#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --gres=gpu:v100:1

#SBATCH --cpus-per-task=3
#SBATCH --mem=12G
#SBATCH --time=5-00:00:00

#SBATCH --output=adni__%j_stdout.out
#SBATCH --error=adni__%j_stderr.out

#SBATCH --array=1-10

#SBATCH --mail-type=ALL
#SBATCH --mail-user=nithesh.merugu5@gmail.com

module load python/3.6
module load cuda

source $HOME/projects/def-jlevman/nithesh/adni_env/bin/activate

python main.py $SLURM_ARRAY_TASK_ID