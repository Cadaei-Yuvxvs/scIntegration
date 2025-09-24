#!/bin/bash --login
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition=gpu_cuda
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie_1g.10gb:1
#SBATCH --mem=32G
#SBATCH --qos=gpu
#SBATCH --account=a_account

module load cuda

export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES


# need to get 3 variables from call
# singularity image
singularity_image=$1
# $method_script : script that should be executed (depend on method)
method_script=$2
# $param_file : specifies which object, assay etc are used and which parameters of the method
param_file=$3

# Run script
echo "singularity exec --nv "$singularity_image" python3 -u "$method_script" "$param_file
srun singularity exec --nv $singularity_image python3 -u $method_script $param_file


