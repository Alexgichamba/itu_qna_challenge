# Specializing Large Language Models for Telecom Networks Challenge Submission

## Documentation

For detailed information about this project, including methodology, architectural diagrams, and in-depth analysis, please refer to our comprehensive documentation:

📚 [Project Documentation](https://3musketeers.gitbook.io/zindi-itu/)

## System
We have tested the inference on two systems. We encourage you to use either. If not, please adapt accordinly.
-------------------------------------
**Instance Details:**
- **Instance Type:** g6.2xlarge (AWS)
- **Operating System:** Ubuntu with Deep Learning Image

**GPU Information:**
- **GPU Model:** NVIDIA L4
- **NVIDIA-SMI Version:** 535.183.01
- **CUDA Version:** 12.2
- **Total GPU Memory:** 23034 MiB
-------------------------------------
OR
-------------------------------------
**Instance Details:**
- **Instance Type:** g5.2xlarge (AWS)
- **Operating System:** Ubuntu with Deep Learning Image

**GPU Information:**
- **GPU Model:** NVIDIA A10G
- **NVIDIA-SMI Version:** 535.183.01
- **CUDA Version:** 12.2
- **Total GPU Memory:** 23034 MiB
-------------------------------------

Note: Training needs at least an A100

## Setup
setup miniconda and conda env
```shell
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
# answer yes to terms and to automatically setting up Miniconda
# reopen terminal
conda deactivate # exit from base env
conda create -n qna python=3.10
conda activate qna
```

install git LFS
```shell
sudo apt install git-lfs
```

clone repo
```shell
git clone https://github.com/Alexgichamba/itu_qna_challenge.git
cd itu_qna_challenge/
```
install dependencies
```shell
chmod +x install_dependencies.sh
./install_dependencies.sh
```

## Inference
for Phi-2
```shell
python3 phi2_final_submission.py
```

for Falcon7B
```shell
python3 falcon_final_submission.py
```