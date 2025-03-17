# SeRe
SeRe is a security-related code review dataset collected through an active learning-based ensemble classifier. SeRe contains 6,732 security-related review comments gathered from a wide range of programming languages (C, C#, C++, Java and Go). The dataset reflects real-world security-related code review distribution, providing a valuable resource for both researchers and practitioners. SeRe enables the evaluation of review generation approaches and highlights the challenges of generating accurate security-focused feedback.

# Structure
In the `code` folder, we provide:
- `data``: 
    - `sere.jsonl`: Final Dataset SeRe
    - `test.json`: Used for RQ2's ensemble classfier's performance evaluation
    - `val.json`: Validation Set used for training
    - `train.json`: training dataset constructed using iterative active learning
- `data_preparation`: The main logic of crawling and storing GitHub code review data modified on ETCR
- `training`: The training script we use

In the `evaluation` folder, we provide:
- `data`: the representative set used for RQ1 and RQ3's evaluation
- `RQ4`: The generated and ground truth comments in RQ4 by different approaches
    - `old`: The results of CodeReviewer, DISCOREV and LLaMA-Reviewer(Lora and Prefix Version) on the test datasets used by their original paper.
    - `new`: The evaluation results on SeRe.

# Usage
## Models
```
'gemma-2-9b-it', 
'Meta-Llama-3-8B-Instruct', 
'deepseek-7b-chat',
'Ministral-8B-Instruct-2410',
'Qwen2.5-7B-Instruct'
```
Please download the corresponding model checkpoint to the specified folder before training.
## Train
```
python train.py --model_name <model_name> --base_model_dir <dir to put base model checkpoint> --output_dir <dir to put fine-tuned classfier>
```
