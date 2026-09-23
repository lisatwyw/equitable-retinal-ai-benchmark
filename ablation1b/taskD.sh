#!/bin/bash
#SBATCH --account=def-lisat-ab      # Replace with your actual billing account/allocation
#SBATCH --job-name=taskAb           # Descriptive job name
#SBATCH --output=logs/job_%A_%a.out
#SBATCH --error=logs/job_%A_%a.err
#SBATCH --time=05:00:00              # Time limit (hh:mm:ss)           
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Number of tasks/processes
#SBATCH --cpus-per-task=16           # Allocate CPU cores per GPU (recommended: up to 16)
#SBATCH --gres=gpu:h100:1           # Request 1 NVIDIA H100 GPU
#SBATCH --mem=25G                   # Memory required (proportionate to your task)
#SBATCH --array=1-2

module purge
module load StdEnv/2023
module load python/3.11
module load apptainer/1.3.4

# Activate Python virtual environment
source /home/lisat/links/projects/def-lisat-ab/lisat/py311/bin/activate


RES1=("GATED_DL" "HYBRID_LR" "HYBRID_SVM" "HYBRID_REGRESSION")

RES=(980 1568)

# Extract the city corresponding to this task's index
res=${RES[$SLURM_ARRAY_TASK_ID]}
echo $res

cd /project/def-lisat-ab/equitable-retinal-ai-benchmark/ablation1b

# Run program
srun python compare.py --RES1 "$res" --task "D" --fold "${SLURM_ARRAY_TASK_ID}"

# Automatically append resource efficiency stats directly into your .out log file
echo "=== Job Efficiency Report ==="
sacct -j "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --format=JobID,JobName,MaxRSS,ReqMem,State
