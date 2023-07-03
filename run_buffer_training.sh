#!/usr/bin/env bash
#SBATCH -t 7-00:00:00 			# time limit days-hours:minutes:seconds
#SBATCH -J d3_d_t_3
#SBATCH -o ./job_outputs/d3_d_t_3_id%j.out
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-node=A40:1 # GPUs 64GB of RAM; cost factor 1.0

# Set and export parameters
export CODE_SIZE=3
export REPETITIONS=3
## Training settings
export NUM_ITERATIONS=3
export BATCH_SIZE=10
export LEARNING_RATE=0.00001
export MANUAL_SEED=12345
## Benchmark
export BENCHMARK=1
## Buffer settings
export BUFFER_SIZE=10
export REPLACEMENTS_PER_ITERATION=2
# test_size is len(error_rate) * batch_size * test_size
export TEST_SIZE=1
## Graph settings
export NUM_NODE_FEATURES=5
export EDGE_WEIGHT_POWER=2
export M_NEAREST_NODES=6
export USE_CUDA=1
export USE_VALIDATION=1

## IO settings
export JOB_NAME=$SLURM_JOB_NAME
## Load old model:
# export RESUMED_TRAINING_FILE_NAME='surface_code_d7_d_t_11_resume_III_id1166198'

# Load modules using pre-installed packages from the Alvis module tree
module purge
module load PyTorch-Geometric/2.1.0-foss-2021a-PyTorch-1.12.1-CUDA-11.3.1
source venv/bin/activate

# Run training script
python3 $SLURM_SUBMIT_DIR/buffer_training.py

