#!/bin/bash

# Check if conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Anaconda or Miniconda and try again."
    exit
fi

echo "Installing the dependencies using pip..."
pip install -r requirements.txt


echo "Installing faiss-gpu using conda..."
conda install -y -c pytorch -c nvidia -c conda-forge faiss-gpu=1.8.0

# download documents
echo "Downloading the documents..."
wget https://api.zindi.africa/v1/competitions/specializing-large-language-models-for-telecom-networks/files/rel18.rar?auth_token=zus.v1.QzSmPPP.fKH7BoxVqMRLRVW3yFhCC3E6GWzUUY
mv rel18.rar* rel18.rar
sudo apt install unrar
unrar x rel18.rar
mv rel18 data
rm rel18.rar

echo "Installation complete."
