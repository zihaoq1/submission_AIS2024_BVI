import os

with open('labels.txt', 'r') as f:
    lines = f.readlines()

trainlines = []
valilines = []
train_list = os.listdir('D:/Datasets/YouTube_UGC/train')
vali_list = os.listdir('D:/Datasets/YouTube_UGC/validation')
for line in lines:
    video = line.split(',')[0]
    score = float(line.split(',')[-1])
    newline = video + ', -1, -1, ' + str(score) + '\n'
    if video in train_list:
        trainlines.append(newline)
    if video in vali_list:
        valilines.append(newline)

with open('train_labels.txt', 'w') as f:
    f.writelines(trainlines)
with open('validation_labels.txt', 'w') as f:
    f.writelines(valilines)