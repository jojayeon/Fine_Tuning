import json
import torch
from transformers import PreTrainedTokenizerFast, LlamaForSequenceClassification

# 학습된 모델 경로 및 데이터셋 경로
model_path = 'result'
data_path = "C:/Users/USER/Fine_Tuning/PY_Learning/data/reallydata.json"

# Llama 토크나이저 초기화 (PreTrainedTokenizerFast 사용)
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)

# 학습된 모델 로드
model = LlamaForSequenceClassification.from_pretrained(model_path)
model.eval()  # 평가 모드로 설정

# 입력 텍스트를 토크나이즈하고 모델에 전달하는 함수 정의
def predict(text):
    # 텍스트 토큰화 및 텐서 변환
    encoding = tokenizer(
        text,
        return_tensors='pt',
        padding='max_length',
        truncation=True,
        max_length=128
    )
    
    # 모델에 입력으로 전달
    with torch.no_grad():  # 그라디언트 계산 비활성화 (추론 모드)
        outputs = model(
            input_ids=encoding['input_ids'],
            attention_mask=encoding['attention_mask']
        )
    
    # 결과 해석
    logits = outputs.logits
    predicted_label = torch.argmax(logits, dim=1).item()
    
    # 레이블 변환 (0 -> 강아지, 1 -> 고양이)
    return "강아지" if predicted_label == 0 else "고양이"

# 테스트용 데이터
test_texts = [
    "이 강아지는 매우 귀여워요.",
    "고양이는 매우 조용하고 우아해요.",
    "저희 집 강아지는 아주 활발해요.",
    "고양이는 낮잠을 많이 자요."
]

# 각 테스트 텍스트에 대해 예측 수행
for text in test_texts:
    result = predict(text)
    print(f"입력: {text}\n예측 결과: {result}\n")
