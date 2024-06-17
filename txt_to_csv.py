import csv

# 读取txt文件
txt_file_path = 'mr_dataset/mr_labels.txt'
csv_file_path = 'mr_dataset/mr_labels.csv'

csvFile = open(csv_file_path, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

# 打开txt文件
f = open(txt_file_path, 'r', encoding='utf-8')

# 逐行读取txt文件中的内容，并依据间距空格进行csv格式的划分
for line in f:
    csvRow = line.split()
    writer.writerow(csvRow)

print(f"Successfully converted {txt_file_path} to {csv_file_path}.")