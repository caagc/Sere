import os
from transformers import AutoTokenizer, GenerationConfig, AutoConfig, AutoModelForSequenceClassification
def get_model_path(model_name, base_dir, has_suffix=True):
    if not has_suffix:
        return os.path.join(base_dir, model_name)
    else:
        return os.path.join(base_dir, model_name, "checkpoint-0")
    
def freeze_model(model):
    for param in model.parameters():
        param.requires_grad = False
        
    for param in model.score.parameters():
        param.requires_grad = True

    last_two_layers = model.model.layers[-1:]
    for layer in last_two_layers:
        for param in layer.parameters():
            param.requires_grad = True

    return model 

def get_model_parameter(model):
    num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # convert num to M
    num = num / 1000000
    return num

def load_classifier(base_dir, model_name, device_map="auto", state="eval"):
    if state == "train":
        print("load from base model")
        model_path = get_model_path(model_name, base_dir=base_dir, has_suffix=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device_map, num_labels=2, trust_remote_code=True)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model.config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        if model_name == "deepseek-7b-chat":
            model.generation_config = GenerationConfig.from_pretrained(model_path)
            model.generation_config.pad_token_id = model.generation_config.eos_token_id
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if model.config.pad_token_id is None:
            model.config.pad_token_id = model.config.eos_token_id
        model = freeze_model(model)
        print(f"model parameter: {get_model_parameter(model)}M")
        return tokenizer, model
    model_path = get_model_path(model_name, base_dir=base_dir, has_suffix=True)
    print(f"load from refine model {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path, device_map=device_map, num_labels=2, trust_remote_code=True)
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.eos_token_id
    return tokenizer, model

def first_comment(comment):
    # return the first comment whose role is "reviewer"
    if isinstance(comment, list):
        for m in comment:
            if m['role'] == 'reviewer':
                return m['message']
        return comment[0]['message']
    return comment

def apply_classifier_prompt(data):
    comment = data["comment"]
    review = first_comment(comment)
    assert review != ""
    prompt = "\n".join([
            f"Please determine whether the reviewer has explicitly identified security issues or stability issues in the code implementation, or explicitly suggested improvements to enhance the security or stability of the code.",
            f"The code review:",
            "-----",
            f"{review}",
            "-----",
        ])
    return prompt