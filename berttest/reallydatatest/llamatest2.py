import json
import torch
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, filepath):
        # JSON 파일에서 데이터 로드
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # Llama 토크나이저 초기화 (PreTrainedTokenizerFast 사용)
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
        # 패딩 토큰 추가
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']  # 질문과 답변 모두 사용
        label = item['label']
        
        # 텍스트를 토크나이즈하고 텐서로 변환
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),  # 입력 ID
            'attention_mask': encoding['attention_mask'].flatten(),  # 어텐션 마스크
            'labels': torch.tensor(0 if label == "강아지" else 1)  # 레이블
        }

# 데이터셋 경로 설정
data_path = 'data/dataset.json'  # 데이터셋 파일 경로
dataset = CustomDataset(data_path)  # 데이터셋 인스턴스 생성

# 모델 초기화
model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels=2)
model.resize_token_embeddings(len(dataset.tokenizer))  # 임베딩 크기 조정

# 학습 설정
training_args = TrainingArguments(
    output_dir='result',
    per_device_train_batch_size=1,
    num_train_epochs=1,
    logging_dir='logs',
    logging_steps=10,  # 손실 값을 10 스텝마다 로그에 기록
    evaluation_strategy='steps',  # 평가 전략을 스텝으로 변경
    eval_steps=10,  # 평가를 10 스텝마다 수행
    save_total_limit=1,
)

# 로깅 콜백 클래스 정의
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def on_log(self, args, state, control, **kwargs):
        # 손실 값 저장
        if 'loss' in kwargs:
            self.losses.append(kwargs['loss'])
        # 정확도 저장 (정확도가 평가 전략에 따라 제공될 경우)
        if 'eval_accuracy' in kwargs:
            self.accuracies.append(kwargs['eval_accuracy'])

# 로깅 콜백 추가
logging_callback = LoggingCallback()
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,  # 전체 데이터셋 사용
    callbacks=[logging_callback]  # 콜백 추가
)

# 학습 실행
trainer.train()

# 모델 저장
model.save_pretrained('result')  # 결과를 result 폴더에 저장
