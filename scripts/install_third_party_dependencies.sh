#!/bin/bash
CONDA_INSTALL_URL=${CONDA_INSTALL_URL:-"https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"}

source scripts/vars.sh

# Install Miniconda locally
rm -rf lib/conda
rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh
wget -P /tmp \
    "${CONDA_INSTALL_URL}" \
    && bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p lib/conda \
    && rm /tmp/Miniconda3-latest-Linux-x86_64.sh

# Grab conda-only packages
export PATH=lib/conda/bin:$PATH
lib/conda/bin/python3 -m pip install nvidia-pyindex
conda env create --name=${ENV_NAME} -f environment.yml
source scripts/activate_conda_env.sh

echo "Attempting to install FlashAttention"
pip install git+https://github.com/HazyResearch/flash-attention.git@5b838a8bef78186196244a4156ec35bbb58c337d && echo "Installation successful"

# Install DeepMind's OpenMM patch
OPENFOLD_DIR=$PWD
pushd lib/conda/envs/$ENV_NAME/lib/python3.7/site-packages/ \
    && patch -p0 < $OPENFOLD_DIR/tools/openmm.patch \
    && popd

echo "Downloading OpenFold parameters..."
bash scripts/params/Evoformer/download_openfold_params.sh Evoformer/openfold/resources

echo "Downloading AlphaFold parameters..."
bash scripts/params/Evoformer/download_alphafold_params.sh Evoformer/openfold/resources

# Decompress test data
gunzip tests/test_data/alphafold/sample_feats.pickle.gz
