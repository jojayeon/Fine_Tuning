from transformers import pipeline

model_id = "meta-llama/Llama-3.2-1B"  # 모델 ID

# 파이프라인 생성
pipe = pipeline(
    "text-generation", 
    model=model_id
)

# 질문 입력
question = "고양이에 대해서 알려줄래"  # 질문 내용

# 모델에 질문하고 응답 받기
response = pipe(question, max_new_tokens=50)  # max_new_tokens를 사용해 최대 생성 토큰 수 조절

# 응답 출력
print(response[0]['generated_text'])
