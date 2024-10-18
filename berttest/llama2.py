from transformers import LlamaForCausalLM, LlamaTokenizer

# 사용할 LLaMA 모델의 이름
model_name = "meta-llama/Llama-3.2-1B"  # 필요한 경우 모델 이름을 적절하게 수정하세요.

# 모델과 토크나이저 로드 (다운로드)
tokenizer = LlamaTokenizer.from_pretrained(model_name, legacy=False)
model = LlamaForCausalLM.from_pretrained(model_name)

print("모델과 토크나이저가 성공적으로 다운로드되었습니다.")
