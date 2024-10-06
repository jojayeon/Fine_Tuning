import json
import torch
import os
import psutil
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from transformers.trainer_callback import TrainerCallback

# GPU 사용 가능 여부 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 메모리 및 디스크 사용량 모니터링을 위한 콜백 클래스
class ResourceMonitorCallback(TrainerCallback):
    def __init__(self, memory_limit=90, disk_limit=90):
        self.memory_limit = memory_limit
        self.disk_limit = disk_limit

    def on_step_end(self, args, state, control, **kwargs):
        memory_percent = psutil.virtual_memory().percent
        disk_percent = psutil.disk_usage('/').percent

        if memory_percent > self.memory_limit:
            print(f"메모리 사용량이 {self.memory_limit}%를 초과했습니다. 현재 사용량: {memory_percent}%")
            time.sleep(10)  # 10초 대기

        if disk_percent > self.disk_limit:
            print(f"디스크 사용량이 {self.disk_limit}%를 초과했습니다. 현재 사용량: {disk_percent}%")
            time.sleep(10)  # 10초 대기

# 모델 및 토크나이저 불러오기
model_id = "meta-llama/Llama-3.2-1B"
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
except Exception as e:
    print(f"모델 로딩 중 오류 발생: {e}")
    exit(1)

# 패딩 토큰 추가
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# 모델의 임베딩 레이어 크기 조정
model.resize_token_embeddings(len(tokenizer))

file_path = "C:/Users/USER/Fine_Tuning/PY_Learning/data/data.json"

# 학습 데이터 로드
try:
    with open(file_path, 'r', encoding='utf-8') as f:
        train_data = json.load(f)
except FileNotFoundError:
    print(f"학습 데이터 파일을 찾을 수 없습니다: {file_path}")
    exit(1)

# 데이터셋 전처리
def preprocess_function(examples):
    inputs = examples['input']
    targets = examples['output']
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=128, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# 데이터셋 생성
dataset = Dataset.from_dict({"input": [item["input"] for item in train_data],
                            "output": [item["output"] for item in train_data]})

# 학습 데이터와 평가 데이터 분할
train_test_dataset = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset = train_test_dataset['train'].map(preprocess_function, batched=True)
eval_dataset = train_test_dataset['test'].map(preprocess_function, batched=True)

# Trainer를 위한 인수 설정
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    eval_steps=100,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=5,
    per_device_eval_batch_size=5,
    num_train_epochs=3,
    gradient_accumulation_steps=4,  # 메모리 사용량 관리를 위해 증가
    dataloader_num_workers=0,
    load_best_model_at_end=True,
    fp16=True if torch.cuda.is_available() else False,  # GPU 사용 가능 시 16비트 연산 사용
)

# Trainer 객체 생성
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[ResourceMonitorCallback()]  # 리소스 모니터링 콜백 추가
)

# 학습 시작
trainer.train()

# 모델 저장
model.save_pretrained("./one/my_model")
tokenizer.save_pretrained("./one/my_model")

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()