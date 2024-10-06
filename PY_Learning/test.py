import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 모델과 토크나이저 로드
model_path = "./my_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# pad_token과 eos_token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_response(prompt, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            top_k=50,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            no_repeat_ngram_size=3,  # 반복 방지
            early_stopping=True  # 조기 종료
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 입력 프롬프트 제거
    response = response.replace(prompt, "").strip()
    
    # 첫 번째 문장 또는 단락만 반환
    sentences = response.split('.')
    if len(sentences) > 1:
        return sentences[0] + '.'
    else:
        return response

# 경고 메시지 숨기기
import warnings
warnings.filterwarnings("ignore")

# 대화 루프
print("모델과 대화를 시작합니다. 종료하려면 'quit'를 입력하세요.")
while True:
    user_input = input("사용자: ")
    if user_input.lower() == 'quit':
        print("대화를 종료합니다.")
        break

    response = generate_response(user_input)
    print("모델:", response)