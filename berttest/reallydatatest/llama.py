import json
import torch
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import TrainerCallback

# CustomDataset 클래스를 정의하여 데이터셋을 처리
class CustomDataset(Dataset):
    def __init__(self, filepath):
        # JSON 파일을 읽어 데이터셋을 초기화
        with open(filepath, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        # 토크나이저를 불러오고 패딩 토큰 추가
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def __len__(self):
        # 데이터셋의 크기를 반환
        return len(self.data)

    def __getitem__(self, idx):
        # 인덱스에 해당하는 데이터 항목을 반환
        item = self.data[idx]
        text = item['text']  # 질문 텍스트
        label = item['label']  # 정답 레이블
        
        # 텍스트를 토큰화하고 텐서로 변환
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=128)
        
        return {
            'input_ids': encoding['input_ids'].flatten(),  # 입력 ID
            'attention_mask': encoding['attention_mask'].flatten(),  # 어텐션 마스크
            'labels': torch.tensor(0 if label == "강아지" else 1)  # 레이블을 정수로 변환
        }

# 데이터셋 경로 설정
data_path = "C:/Users/Administrator/jojayeon/Fine_Tuning/PY_Learning/data/reallydata.json"
dataset = CustomDataset(data_path)  # CustomDataset 객체 생성

# 모델 초기화
model = LlamaForSequenceClassification.from_pretrained("meta-llama/Llama-3.2-1B", num_labels=2)
model.resize_token_embeddings(len(dataset.tokenizer))  # 모델의 임베딩 크기 조정

# gpu 사용가능하면 사용하라고 적어주고 

# GPU가 사용 가능한지 확인하고 모델을 해당 장치로 이동
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# gpu 사용을 하고 있오 어떤 모델을 사용하고 있는지 확인

if torch.cuda.is_available():
    print(f"GPU를 사용하고 있습니다: {torch.cuda.get_device_name(0)}")
else:
    print("GPU를 사용할 수 없습니다. CPU를 사용합니다.")
    
    
if torch.cuda.is_available():
    print(f"총 GPU 수: {torch.cuda.device_count()}")
    print(f"현재 GPU 인덱스: {torch.cuda.current_device()}")
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    print(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()} bytes")
    
    
# 학습 설정
training_args = TrainingArguments(
    output_dir='result',  # 결과를 저장할 디렉토리
    per_device_train_batch_size=1,  # 각 GPU에서의 배치 크기
    per_device_eval_batch_size=1,  # 평가 시 배치 크기
    num_train_epochs=5,  # 학습 에폭 수
    learning_rate=5e-5,  # 학습률
    weight_decay=0.01,  # 가중치 감쇠
    gradient_accumulation_steps=4,  # 그래디언트 축적 스텝 수
    logging_dir='logs',  # 로깅 파일을 저장할 디렉토리
    logging_steps=100,  # 로깅 빈도 (스텝 수)
    eval_strategy='steps',  # 평가 전략
    eval_steps=100,  # 평가 빈도 (스텝 수)
    load_best_model_at_end=True,  # 최적 모델 로드 설정
    metric_for_best_model='accuracy'  # 최적 모델 기준
)

# 로깅 콜백 클래스 정의
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []  # 손실 값을 저장할 리스트
        self.accuracies = []  # 정확도를 저장할 리스트

    def on_log(self, args, state, control, **kwargs):
        # 로깅 시 손실 값과 정확도를 저장하고 출력
        if 'loss' in kwargs:
            loss = kwargs['loss']
            self.losses.append(loss)  # 손실 값을 리스트에 추가
            print(f"Step {state.global_step}: Loss - {loss}")  # 콘솔에 손실 값 출력

        if 'eval_accuracy' in kwargs:
            accuracy = kwargs['eval_accuracy']
            self.accuracies.append(accuracy)  # 정확도를 리스트에 추가
            print(f"Step {state.global_step}: Accuracy - {accuracy}")  # 콘솔에 정확도 출력


    # 그래프 만들기 
    def on_train_end(self, args, state, control, **kwargs):
        # 학습 종료 후 그래프 생성
        plt.figure(figsize=(10, 5))  # 그래프 크기 설정

        # 손실 값 그래프
        plt.subplot(1, 2, 1)  # 1행 2열의 첫 번째 서브플롯
        plt.plot(self.losses, label='Training Loss')  # 손실 값 플로팅
        plt.title('Training Loss Over Steps')  # 그래프 제목
        plt.xlabel('Step')  # x축 레이블
        plt.ylabel('Loss')  # y축 레이블
        plt.legend()  # 범례 추가

        # 정확도 그래프
        plt.subplot(1, 2, 2)  # 1행 2열의 두 번째 서브플롯
        plt.plot(self.accuracies, label='Validation Accuracy')  # 정확도 플로팅
        plt.title('Validation Accuracy Over Steps')  # 그래프 제목
        plt.xlabel('Step')  # x축 레이블
        plt.ylabel('Accuracy')  # y축 레이블
        plt.legend()  # 범례 추가

        plt.show()  # 그래프 출력

# 로깅 콜백 추가
logging_callback = LoggingCallback()  # 로깅 콜백 객체 생성
trainer = Trainer(
    model=model,  # 모델
    args=training_args,  # 학습 설정
    train_dataset=dataset,  # 학습 데이터셋
    callbacks=[logging_callback]  # 콜백 추가
)

# 학습 실행
trainer.train()

# 학습 완료 후 모델 저장
model.save_pretrained('result')  # 모델을 지정한 경로에 저장
