from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()

origins = [
    "http://localhost:3000",  # 개발 중인 로컬 주소
    "https://myaillama.s3.ap-northeast-2.amazonaws.com",  # S3 호스팅 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 도메인에서 접근 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 모델과 토크나이저 로드
model_path = "C:/Users/USER/Fine_Tuning/PY_Learning/two/my_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# pad_token과 eos_token 설정
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.eos_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 질문 모델 데이터 클래스를 정의
class Question(BaseModel):
    question: str

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
            early_stopping=True,  # 사용할 경우 num_beams 값을 1보다 큰 값으로 설정해야 함
            num_beams=5  # 예: 5로 설정
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.replace(prompt, "").strip()

    # 한 개의 문자열로 반환
    return response

@app.post('/api/question')
async def answer_question(question: Question):
    # 모델을 사용하여 질문에 대한 답변 생성
    answer = generate_response(question.question)
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
