#!/usr/bin/env bash
#SBATCH --job-name=channel_trans
#SBATCH --account=gts-apm7                 # <-- your PACE account
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL

# Safety: avoid -u due to site bashrc, but keep -e for fail-fast
set -eo pipefail

# ===== user settings =====
CONDA_ENV="torch_legacy"                      # your conda env
SCRIPT="/storage/home/hcoda1/4/xchen920/scratch/project/evaluate.py"   # <-- path to the Python file
# =========================

echo "== Job info =="; date; hostname

module purge || true
module load anaconda3/2022.05 2>/dev/null || true

# Activate conda without sourcing user/site bashrcs that may break nounset
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
  . "$HOME/anaconda3/etc/profile.d/conda.sh"
fi
conda activate "$CONDA_ENV"

echo "== GPU check =="
nvidia-smi || { echo "No GPU visible"; exit 1; }

# Go to project directory (optional)
cd "$(dirname "$SCRIPT")"

echo "== Launch =="
srun -u python "$(basename "$SCRIPT")" \

echo "== Done =="; date
