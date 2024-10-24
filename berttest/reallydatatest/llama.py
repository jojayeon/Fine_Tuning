import json
import torch
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import TrainerCallback


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
# data_path = "C:/Users/Administrator/jojayeon/Fine_Tuning/PY_Learning/data/reallydata.json"  # 데이터셋 파일 경로
data_path = "C:/Users/USER/Fine_Tuning/PY_Learning/data/reallydata.json"  # 데이터셋 파일 경로
dataset = CustomDataset(data_path)  # 데이터셋 인스턴스 생성

# 모델 초기화
model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels=2)
model.resize_token_embeddings(len(dataset.tokenizer))  # 임베딩 크기 조정

# 학습 설정
training_args = TrainingArguments(
    output_dir='result',
    per_device_train_batch_size=1,  # 배치 크기
    num_train_epochs=5,  # 에폭 수
    learning_rate=5e-5,  # 학습률
    weight_decay=0.01,  # 가중치 감소
    gradient_accumulation_steps=4,  # 그래디언트 누적 스텝
    logging_dir='logs',
    logging_steps=100,  # 100 스텝마다 로깅
    eval_strategy='steps',  # 평가 전략
    eval_steps=100,  # 평가 스텝
    load_best_model_at_end=True,  # 최상의 모델 로드
    metric_for_best_model='accuracy',  # 최상의 모델 판단 기준
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
