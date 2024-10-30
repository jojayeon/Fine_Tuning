import os
import json
import torch
import logging
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from transformers import TrainerCallback
from bitsandbytes.optim import AutoOptimizer
from bitsandbytes.nn import Linear4bit
from bitsandbytes import AutoOptimizer
from bitsandbytes.optim import Adam8bit

# 로깅 설정
logging.basicConfig(
    filename='training.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger()

# CUDA 메모리 관리 설정
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
torch.cuda.empty_cache()

class CustomDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        label = item['label']
        encoding = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=64)
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(0 if label == "강아지" else 1)
        }

def load_data_batch(filepath, start_idx, batch_size):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data[start_idx:start_idx+batch_size]

def train_in_batches(model, tokenizer, data_path, batch_size=100, num_epochs=3):
    with open(data_path, 'r', encoding='utf-8') as f:
        full_data = json.load(f)
    
    total_samples = len(full_data)
    num_batches = (total_samples + batch_size - 1) // batch_size

    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        for batch in range(num_batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, total_samples)
            
            batch_data = full_data[start_idx:end_idx]
            dataset = CustomDataset(batch_data, tokenizer)
            
            training_args = TrainingArguments(
                output_dir=f'result_epoch_{epoch}_batch_{batch}',
                per_device_train_batch_size=1,
                num_train_epochs=1,
                learning_rate=1e-5,
                weight_decay=0.01,
                logging_dir='logs',
                logging_steps=10,
                save_strategy='steps',
                save_steps=100,
                save_total_limit=1,
                fp16=True,
                report_to="none",
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset,
            )
            
            try:
                trainer.train()
                logger.info(f"Completed training for epoch {epoch + 1}, batch {batch + 1}/{num_batches}")
            except Exception as e:
                logger.error(f"Error during training: {e}")
                raise e

    # 최종 모델 저장
    try:
        model.save_pretrained('final_result')
        logger.info("Final model saved successfully.")
    except Exception as e:
        logger.error(f"Error saving final model: {e}")
        raise e

# 메인 실행 부분
data_path = "C:/Users/Administrator/jojayeon/Fine_Tuning/PY_Learning/data/reallydata2.json"

try:
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # INT4 양자화를 적용한 모델 로드
    model = LlamaForSequenceClassification.from_pretrained(
        "meta-llama/Llama-3.2-1B",
        num_labels=2,
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.resize_token_embeddings(len(tokenizer))
    
    # 모든 Linear 레이어를 4비트 양자화된 버전으로 변경
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            model._modules[name] = Linear4bit(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
                compute_dtype=torch.float16
            )
    
    logger.info("Model initialized successfully with INT4 quantization.")
except Exception as e:
    logger.error(f"Error initializing model: {e}")
    raise e

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

if torch.cuda.is_available():
    logger.info(f"GPU 사용 중: {torch.cuda.get_device_name(0)}")
    logger.info(f"총 GPU 수: {torch.cuda.device_count()}")
    logger.info(f"현재 GPU 인덱스: {torch.cuda.current_device()}")
    logger.info(f"현재 GPU 이름: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    logger.info(f"현재 GPU 메모리 사용량: {torch.cuda.memory_allocated()} bytes")

train_in_batches(model, tokenizer, data_path, batch_size=100, num_epochs=3)