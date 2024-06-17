from transformers import BertTokenizer, BertForSequenceClassification, AdamW
import torch
import re
import pandas as pd
from tqdm import tqdm


# 清理文本的函数
def clear_character(sentence):
    pattern1 = '[^a-zA-Z0-9\u4e00-\u9fa5\s]'  # 保留中英文字符、数字和空格
    line = re.sub(pattern1, '', sentence)  # 移除其他字符
    return line.strip()  # 去除首尾空白字符


# 加载BERT的tokenizer和预训练模型
tokenizer = BertTokenizer.from_pretrained('models--bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('models--bert-base-uncased', num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 加载和清理数据
with open('mr_dataset/mr.txt', 'r', encoding='utf-8', errors='replace') as file:
    lines = file.readlines()

cleaned_lines = [clear_character(line) for line in lines]
data = pd.read_csv('mr_dataset/mr_labels.csv')

print(len(cleaned_lines))
print(len(data))
# 检查数据长度是否匹配
assert len(cleaned_lines) == len(data), "Number of reviews and labels must match"

# 定义优化器和损失函数
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = torch.nn.CrossEntropyLoss()

# 训练模型
model.train()
num_epochs = 3
train_id = 7107

for epoch in range(num_epochs):
    total_loss = 0
    for i in tqdm(range(train_id)):
        inputs = tokenizer(cleaned_lines[i], padding=True, truncation=True, return_tensors='pt').to(device)
        labels = torch.tensor([data['label'][i]]).to(device)  # 假设标签列名为 'label'

        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f'Epoch: {epoch + 1}, Loss: {total_loss / train_id}')

# 保存模型权重
model.save_pretrained('./model_weights')

model.eval()
correct = 0
total = len(cleaned_lines)

with torch.no_grad():
    for i in tqdm(range(total)):
        if i <= train_id:
            continue
        inputs = tokenizer(cleaned_lines[i], padding=True, truncation=True, return_tensors='pt').to(device)
        labels = torch.tensor([data['label'][i]]).to(device)  # 假设测试标签列名为 'label'

        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        correct += (predicted == labels).sum().item()
        print(predicted)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')

