from transformers import PreTrainedTokenizerFast , LlamaForCausalLM
import torch

# 모델과 토크나이저 로드
model_name = "meta-llama/Llama-3.2-1B"  # 사용하고자 하는 모델 이름
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)
model = LlamaForCausalLM.from_pretrained(model_name)

# 모델을 평가 모드로 전환
model.eval()

# 테스트 입력 텍스트
input_text = "인공지능에 대해 어떻게 생각하시나요?"

# pad_token을 eos_token으로 설정 # 아래 input에서 패팅을 사용하려면 패팅을 지정해줘야하기떄무네 설정
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# 입력 텍스트를 토큰화
inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)

# 모델 예측
with torch.no_grad():
    outputs = model.generate( inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=100)  # 최대 출력 길이 설정

# 출력 토큰을 텍스트로 변환
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("입력 텍스트:", input_text)
print("모델 출력:", output_text)
