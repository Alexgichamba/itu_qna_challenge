# Specializing Large Language Models for Telecom Networks Challenge Submission

## Documentation

For detailed information about this project, including methodology, architectural diagrams, and in-depth analysis, please refer to our comprehensive documentation:

📚 [Project Documentation](https://3musketeers.gitbook.io/zindi-itu/)

## Setup

```shell
conda create -n itu_qna python=3.10.13
conda activate itu_qna
pip install -r requirements.txt
```

## Ablation Studies

### RAG Hyperparameters

#### Chunk Size: 50
| k   | Accuracy (%) |
|-----|--------------|
| 1   | 69.9         |
| 3   | 71.3         |
| 10  | 73.7         |

#### Chunk Size: 100
| k   | Accuracy (%) |
|-----|--------------|
| 1   | 72.6         |
| 3   | 72.9         |
| 6   | 75.4         |
| 7   | 75.6         |

#### Chunk Size: 150
| k   | Accuracy (%) |
|-----|--------------|
| 5   | 75.6         |
| 6   | 76.2         |
| 7   | 77.0         |
| 8   | 77.0         |
| 9   | 76.5         |
| 10  | 76.7         |
| 12  | 75.1         |

#### Chunk Size: 180
| k   | Accuracy (%) |
|-----|--------------|
| 1   | 74.3         |
| 2   | 73.7         |
| 3   | 75.6         |
| 4   | 75.1         |

#### Chunk Size: 250
| k   | Accuracy (%) |
|-----|--------------|
| 1   | 73.22        |
| 3   | 73.22        |
| 5   | 71.1         |

### BM25

#### Chunk Size: 150
| k   | Accuracy (%) |
|-----|--------------|
| 3   | 68.3         |
| 10  | 67.2         |

#### Chunk Size: 300
| k   | Accuracy (%) |
|-----|--------------|
| 3   | 67.2         |

## Training and Inference

### Hugging Face Login
Before training, log in to Hugging Face:
```shell
huggingface-cli login
```

### Finetuning Phi-2 on Questions
```shell
python3 finetuning/trainer_qns.py --num_epochs <num_epochs>
```

### Inference with Phi-2
```shell
python3 phi2_inference.py --model_name <model_name_on_HF> --adapter_path <check_point>
```