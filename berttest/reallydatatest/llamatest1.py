import torch
from transformers import PreTrainedTokenizerFast, LlamaForCausalLM, Trainer, TrainingArguments
from datasets import load_dataset, Dataset
import json

# JSON 파일 경로 지정
json_file_path = "C:/Users/Administrator/jojayeon/Fine_Tuning/PY_Learning/data/reallydata.json"

# JSON 파일 불러오기
with open(json_file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# 텍스트 데이터를 Dataset으로 변환
dataset = Dataset.from_list([{"text": item["text"]} for item in data])

# LLaMA 3.2 1B 모델과 토크나이저 불러오기
model_name = "meta-llama/Llama-3.2-1B"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)
print("Vocabulary size:", tokenizer.vocab_size)

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 데이터셋을 모델 입력 형식으로 인코딩
def preprocess_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

encoded_dataset = dataset.map(preprocess_function, batched=True)
print(encoded_dataset[0])
# 학습 설정
training_args = TrainingArguments(
    output_dir="./result",            # 결과를 저장할 경로를 'result'로 지정
    per_device_train_batch_size=1,    # 성능 부하를 줄이기 위해 배치 크기를 작게 설정
    num_train_epochs=100,              # 각 텍스트당 100번의 학습을 진행
    logging_dir='./logs',             # 로그 저장 경로
    logging_steps=10,                 # 로그를 얼마나 자주 출력할지 설정
    save_steps=1000,                  # 체크포인트 저장 빈도
    save_total_limit=2,               # 최대 저장 체크포인트 수
)

# Trainer 인스턴스 생성
trainer = Trainer(
    model=model,                      # 학습할 모델
    args=training_args,               # 학습 설정
    train_dataset=encoded_dataset     # 학습에 사용할 데이터셋
)

# 학습 시작
trainer.train()

# 학습 완료 후 모델 저장
trainer.save_model("./result")

# 토크나이저 저장 (나중에 다시 사용할 수 있도록)
tokenizer.save_pretrained("./result")
