set -ex

folder=./

# prepare conda environment
conda create -n fairseq python==3.7 --yes
source activate fairseq
cd code/tools/fairseq
pip install --editable ./
cd ~-
pip install -r code/requirements.txt

# download corpus
cd $folder
sh code/script_download_data.sh

