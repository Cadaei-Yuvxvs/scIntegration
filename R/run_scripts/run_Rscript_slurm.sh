#!/bin/bash --login
#SBATCH --ntasks=1
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=6G
#SBATCH --partition=general
#SBATCH --qos=normal
#SBATCH --account=a_account


# need to get 3 variables from call
# singularity image
singularity_image=$1
# $method_script : script that should be executed (depend on method)
method_script=$2
# $param_file : specifies which object, assay etc are used and which parameters of the method
param_file=$3

# Run script
echo "singularity exec "$singularity_image" Rscript "$method_script" "$param_file
srun singularity exec $singularity_image Rscript $method_script $param_file
