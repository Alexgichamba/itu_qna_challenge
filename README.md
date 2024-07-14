# Specializing Large Language Models for Telecom Networks challenge submission


```shell
conda create -n itu_qna python=3.10.13
conda activate itu_qna
pip install -r requirements.txt
```

## Ablations
### RAG hyperparams
when size=50
|k   | Acc(%)|
|----|-------|
|1   |69.9   |
|3   |71.3   |
|10  |73.7   |

when size=100
|k   | Acc(%)|
|----|-------|
|1   |72.6   |
|3   |72.9   |
|6   |75.4   |
|7   |75.6   |

when size=150
|k   | Acc(%)|
|----|-------|
|5   |75.6   |
|6   |76.2   |
|7   |77.0   |
|8   |77.0   |
|9   |76.5   |
|10  |76.7   |
|12  |75.1   |

when size=180
|k   | Acc(%)|
|----|-------|
|1   |74.3   |
|2   |73.7   |
|3   |75.6   |
|4   |75.1   |

when size=250
|k   | Acc(%)|
|----|-------|
|1   |73.22  |
|3   |73.22  |
|5   |71.1   |

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
python3 phi2_inference.py --model_name <model_name_on_HF> --adapter_path <check_point>
```