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








echo "Installation complete."
