import torch
from transformers import LlamaForSequenceClassification, PreTrainedTokenizerFast

def load_model_and_tokenizer(model_path):
    # 모델과 토크나이저 불러오기
    model = LlamaForSequenceClassification.from_pretrained(model_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained("meta-llama/Llama-3.2-1B")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    return model, tokenizer

def predict(model, tokenizer, text):
    # 입력 텍스트 토큰화
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=64)
    
    # GPU 사용 가능 시 GPU로 이동
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 예측 수행
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 결과 해석
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    
    return "강아지" if predicted_class == 0 else "고양이"

def main():
    # 모델 경로 설정
    model_path = "result"  # 학습된 모델이 저장된 경로

    # 모델과 토크나이저 불러오기
    model, tokenizer = load_model_and_tokenizer(model_path)

    # 테스트할 텍스트 목록
    test_texts = [
        "공원에서 뛰어놀고 있어요.",
        "창틀에서 낮잠을 자고 있습니다.",
        "반려동물과 함께 산책하는 것은 정말 즐거워요.",
        "매일 아침 산책을 갑니다.",
        "독립적인 성격을 가지고 있어요."
    ]

    # 각 텍스트에 대해 예측 수행
    for text in test_texts:
        prediction = predict(model, tokenizer, text)
        print(f"텍스트: {text}")
        print(f"예측 결과: {prediction}\n")

if __name__ == "__main__":
    main()