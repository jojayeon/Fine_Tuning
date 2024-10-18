from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

token = tokenizer("안녕하세요. 파인 튜닝 공부 중입니다")
print(token)