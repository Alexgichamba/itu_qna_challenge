# Specializing Large Language Models for Telecom Networks challenge submission


```shell
conda create -n itu_qna python=3.11
conda activate itu_qna
pip install -r requirements.txt
```

## Training
First login to HF
```shell
huggingface-cli login
```
### Finetuning Phi-2 on the questions
```shell
python3 finetuning/trainer_qns.py --num_epochs <num_epochs>
```

### Inference with Phi-2
```shell
python3 phi2_inference.py --model_name "alexgichamba/phi-2-finetuned-qa-lora-r32-a16" --adapter_path save_phi2_ft_lora/checkpoint-300/
```