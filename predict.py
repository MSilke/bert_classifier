from transformers import BertTokenizer
import torch
from transformers import BertForSequenceClassification, AdamW

# 参数设置
BATCH_SIZE = 16
MAX_LEN = 128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer用于部署
model = BertForSequenceClassification.from_pretrained('./model_weights')
tokenizer = BertTokenizer.from_pretrained('./model_weights')
model.to(device)

# 示例推理函数
def predict(text):
    model.eval()
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, predicted = torch.max(outputs.logits, 1)
        return predicted.item()

# 测试推理函数
sample_text = "OHHH! the movie is not bad, but sometimes it make me mad!"

print(f'Text: {sample_text}, Predicted label: {predict(sample_text)}')