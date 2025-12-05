#!/bin/bash
#SBATCH --nodes=1               # node count
#SBATCH -p gpu --gres=gpu:2     # number of gpus per node
#SBATCH --ntasks-per-node=1     # total number of tasks across all nodes
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH -t 06:30:00             # total run time limit (HH:MM:SS)
#SBATCH --mem=64000MB           # INCREASED from 16GB to 32GB
#SBATCH --job-name='MyGO'
#SBATCH --output=slurm_logs/R-%x.%j.out
#SBATCH --error=slurm_logs/R-%x.%j.err
# Force unbuffered output
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
# TensorFlow GPU settings
export TF_FORCE_GPU_ALLOW_GROWTH=true
module purge
unset LD_LIBRARY_PATH
export APPTAINER_BINDPATH="/oscar/home/$USER,/oscar/scratch/$USER,/oscar/data"
# Use the correct pre-built container path (note: x86_64.d not x86_64)
CONTAINER_PATH="/oscar/runtime/software/external/ngc-containers/tensorflow.d/x86_64.d/tensorflow-24.03-tf2-py3.simg"
EXEC_PATH="srun apptainer exec --nv"
echo ""
echo "=========================================="
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo ""

echo "GPU Information (from host):"
nvidia-smi
echo ""

# echo "GPU Information (inside container):"
# $EXEC_PATH $CONTAINER_PATH nvidia-smi
# echo ""

# echo "TensorFlow GPU Detection:"
# $EXEC_PATH $CONTAINER_PATH python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPUs detected:', len(tf.config.list_physical_devices('GPU'))); print('GPU devices:', tf.config.list_physical_devices('GPU'))"
# echo ""

module load miniconda3/23.11.0s
conda activate mygo
module load ffmpeg
which ffmpeg
ffmpeg -version

python -c "import tensorflow as tf; print('TF version:', tf.__version__); print('Built with CUDA:', tf.test.is_built_with_cuda()); print('GPUs detected:', len(tf.config.list_physical_devices('GPU'))); print('GPU devices:', tf.config.list_physical_devices('GPU'))"

echo "=========================================="
echo "Installing dependencies"
echo "=========================================="
echo ""

# $EXEC_PATH $CONTAINER_PATH pip install --user --no-cache-dir tqdm wandb

echo ""
echo "=========================================="
echo "Starting main Python script at $(date)"
echo "=========================================="
echo ""

cd "${SLURM_SUBMIT_DIR}" || exit 1
echo "Working directory: $(pwd)"
echo ""

# $EXEC_PATH $CONTAINER_PATH python -u train.py
python3 -u train.py

EXIT_CODE=$?

echo ""
echo "=========================================="
echo "Python script finished at $(date)"
echo "Exit code: $EXIT_CODE"
echo "=========================================="
exit $EXIT_CODE
