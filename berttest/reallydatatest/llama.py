import os
import json
import torch
import logging
import matplotlib.pyplot as plt
from transformers import PreTrainedTokenizerFast, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import TrainerCallback
from transformers import AutoModelForSequenceClassification, BitsAndBytesConfig

# 로깅 설정
logging.basicConfig(
    filename='training.log',  # 로그 파일 이름
    level=logging.DEBUG,  # 로그 레벨 설정
    format='%(asctime)s - %(levelname)s - %(message)s',  # 로그 메시지 형식
)
logger = logging.getLogger()

# 메모리 최적화 옵션 추가 (CUDA 메모리 관리)
# 둘중 하나 선택해서 사용하기
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

torch.cuda.empty_cache()  # GPU 메모리 캐시 초기화

# 사용해보았지만 적용하기 전에 멈춤 
# torch.cuda.set_per_process_memory_fraction(0.8, 0) # gpu사용량 50%

# CustomDataset 클래스를 정의하여 데이터셋을 처리
class CustomDataset(Dataset):
    def __init__(self, filepath):
        try:
            # JSON 파일을 읽어 데이터셋을 초기화
            with open(filepath, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            # 토크나이저를 불러오고 패딩 토큰 추가
            self.tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            logger.info("Dataset and tokenizer initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing dataset: {e}")
            raise e

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            item = self.data[idx]
            text = item['text']
            label = item['label']
            encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(0 if label == "강아지" else 1)
            }
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            raise e

# 데이터셋 경로 설정
# data_path = "C:/Users/USER/Fine_Tuning/PY_Learning/data/reallydata.json"
data_path = "C:/Users/Administrator/jojayeon/Fine_Tuning/PY_Learning/data/reallydata2.json"
dataset = CustomDataset(data_path)

# INT4 양자화 설정
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# 모델 초기화 및 INT4 양자화 적용
try:
    model = AutoModelForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        num_labels=2,
        quantization_config=quantization_config
    )
    model.config.pad_token_id = dataset.tokenizer.pad_token_id
    model.resize_token_embeddings(len(dataset.tokenizer))
    logger.info("Model initialized and quantized successfully.")
except Exception as e:
    logger.error(f"Error initializing and quantizing model: {e}")
    raise e

# GPU가 사용 가능한지 확인하고 모델을 해당 장치로 이동
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

# GPU 사용 정보 확인
if torch.cuda.is_available():
    logger.info(f"GPU 사용 중: {torch.cuda.get_device_name(0)}")
    logger.info(f"총 GPU 수: {torch.cuda.device_count()}")
    logger.info(f"현재 GPU 인덱스: {torch.cuda.current_device()}")
    logger.info(f"현재 GPU 이름: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()} bytes")

# 학습 설정
training_args = TrainingArguments(
    output_dir='result',  # 학습 결과와 모델이 저장될 경로
    per_device_train_batch_size=1,  # 각 GPU당 학습 배치 크기 (메모리 사용을 줄이기 위해 소규모로 설정)
    per_device_eval_batch_size=1,  # 각 GPU당 평가 배치 크기
    num_train_epochs=3,  # 에폭 수를 줄여 초기 학습 시 안정적으로 진행 (필요시 증가 가능)
    gradient_accumulation_steps=4,  # 작은 배치 크기를 보완하기 위해 그래디언트 누적 단계 수를 증가시킴
    learning_rate=1e-5,  # 학습률을 낮춰 안정적 학습 유도 (큰 값일 경우 불안정한 학습 가능)
    weight_decay=0.01,  # 가중치 감소를 통한 과적합 방지
    logging_dir='logs',  # 로그 파일이 저장될 경로
    logging_steps=10,  # 로그를 더 자주 남기도록 설정 (학습의 진행 상황을 빠르게 파악 가능)
    eval_strategy='steps',  # 일정 간격으로 평가 진행
    eval_steps=100,  # 평가 간격 (로그 단계와 맞춤)
    save_strategy='steps',  # 학습 도중 주기적으로 모델을 저장
    save_steps=100,  # 주기적으로 모델을 저장하는 단계
    save_total_limit=3,  # 저장되는 모델의 수를 제한하여 디스크 용량 절약
    load_best_model_at_end=True,  # 학습 종료 후 가장 성능이 좋은 모델을 로드
    metric_for_best_model='accuracy',  # 최적 모델 판단 기준을 정확도로 설정
    fp16=False,  # 16비트 부동 소수점 사용 (메모리와 연산 효율을 위해 사용)
    no_cuda=False,  # CUDA 사용 여부 (GPU가 있는 경우 자동으로 사용하도록 설정)
    report_to="none",  # 기본 보고 설정 비활성화 (필요 시 WandB, TensorBoard 등 사용 가능)
)

# 로깅 콜백 클래스 정의
class LoggingCallback(TrainerCallback):
    def __init__(self):
        self.losses = []
        self.accuracies = []

    def on_log(self, args, state, control, **kwargs):
        if 'loss' in kwargs:
            loss = kwargs['loss']
            self.losses.append(loss)
            logger.info(f"Step {state.global_step}: Loss - {loss}")
        
        if 'eval_accuracy' in kwargs:
            accuracy = kwargs['eval_accuracy']
            self.accuracies.append(accuracy)
            logger.info(f"Step {state.global_step}: Accuracy - {accuracy}")

    def on_train_end(self, args, state, control, **kwargs):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Training Loss')
        plt.title('Training Loss Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies, label='Validation Accuracy')
        plt.title('Validation Accuracy Over Steps')
        plt.xlabel('Step')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

# 로깅 콜백 추가
logging_callback = LoggingCallback()

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    callbacks=[logging_callback]
)

# 학습 실행
try:
    trainer.train()
    logger.info("Training completed successfully.")
except Exception as e:
    logger.error(f"Error during training: {e}")
    raise e

# 학습 완료 후 모델 저장
try:
    model.save_pretrained('result')
    logger.info("Model saved successfully.")
except Exception as e:
    logger.error(f"Error saving model: {e}")
    raise e
