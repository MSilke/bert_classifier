import re
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from transformers import BertForSequenceClassification, AdamW
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns

# 清理文本的函数
def clear_character(sentence):
    pattern1 = '[^a-zA-Z0-9\u4e00-\u9fa5\s]'  # 保留中英文字符、数字和空格
    line = re.sub(pattern1, '', sentence)  # 移除其他字符
    return line.strip()  # 去除首尾空白字符

# 加载和清理数据
with open('mr_dataset/mr.txt', 'r', encoding='utf-8', errors='replace') as file:
    lines = file.readlines()
cleaned_lines = [clear_character(line) for line in lines]
data = pd.read_csv('mr_dataset/mr_labels.csv')

assert len(cleaned_lines) == len(data), "Number of reviews and labels must match"

# 加载BERT模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型和tokenizer用于部署
model = BertForSequenceClassification.from_pretrained('./model_weights')
tokenizer = BertTokenizer.from_pretrained('./model_weights')
model.to(device)

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(cleaned_lines, data['label'], test_size=0.2, random_state=42)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 参数设置
BATCH_SIZE = 16
MAX_LEN = 128

# 创建数据集和数据加载器
train_dataset = SentimentDataset(train_texts, train_labels.tolist(), tokenizer, MAX_LEN)
val_dataset = SentimentDataset(val_texts, val_labels.tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 模型评估
model.eval()
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for batch in tqdm(val_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]  # 预测为正类的概率
        _, predicted = torch.max(outputs.logits, 1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(predicted.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

# 计算评估指标
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)
auc_roc = roc_auc_score(all_labels, all_probs)

print(f'Test Accuracy: {accuracy * 100:.2f}%')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print(f'AUC-ROC: {auc_roc:.2f}')

# 生成并可视化混淆矩阵
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

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
sample_text = "OHHH! the movie is crazy bad!"
print(f'Text: {sample_text}, Predicted label: {predict(sample_text)}')