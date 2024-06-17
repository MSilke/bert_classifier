import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib
import matplotlib.pyplot as plt
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

# 划分数据集
train_texts, val_texts, train_labels, val_labels = train_test_split(cleaned_lines, data['label'], test_size=0.2, random_state=42)

# 使用TF-IDF向量化文本数据，添加停用词过滤
vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train = vectorizer.fit_transform(train_texts)
X_val = vectorizer.transform(val_texts)

# 检查是否有空的向量
assert X_train.shape[0] == len(train_texts), "Some training documents resulted in empty vectors"
assert X_val.shape[0] == len(val_texts), "Some validation documents resulted in empty vectors"

# 训练SVM模型
svm_model = SVC(kernel='linear', probability=True)
svm_model.fit(X_train, train_labels)

# 评估模型
val_predictions = svm_model.predict(X_val)
val_probabilities = svm_model.predict_proba(X_val)[:, 1]

accuracy = accuracy_score(val_labels, val_predictions)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# 输出详细的分类报告
print(classification_report(val_labels, val_predictions))

# 计算并打印AUC-ROC
auc_roc = roc_auc_score(val_labels, val_probabilities)
print(f'AUC-ROC: {auc_roc:.2f}')

# 生成并可视化混淆矩阵
conf_matrix = confusion_matrix(val_labels, val_predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# 保存模型和向量化器
joblib.dump(svm_model, 'svm_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# 加载模型和向量化器
svm_model = joblib.load('svm_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict(text):
    cleaned_text = clear_character(text)
    vectorized_text = vectorizer.transform([cleaned_text])
    prediction = svm_model.predict(vectorized_text)
    return prediction[0]

# 测试推理函数
sample_text = "This is a sample review."
print(f'Text: {sample_text}, Predicted label: {predict(sample_text)}')