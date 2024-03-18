#!/bin/bash
mkdir -p $HOME/jobs

BATCH_FILE=$(mktemp -p $HOME/jobs --suffix=.job)

cat > ${BATCH_FILE} <<EOF
#!/bin/bash -x
#SBATCH --job-name=$(echo $1 | cut -d "_" -f 4)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=0-24:00:00
#SBATCH --gpus=1
#SBATCH --mem=150G
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu-a40
#SBATCH --account=${2}

source /mmfs1/gscratch/sciencehub/sebgab/miniconda3/bin/activate
cd /mmfs1/gscratch/sciencehub/sebgab/Dev/StowSAM
conda activate StowSam

python new_data_preprocess.py
exit 0

EOF


sbatch ${BATCH_FILE}
