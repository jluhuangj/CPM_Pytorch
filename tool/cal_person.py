import json
import sys
import os

data_json = '/Users/huangju/Desktop/COCO.json'
w_train_f_txt = '/Users/huangju/Desktop/train_name.txt'
w_val_f_txt = '/Users/huangju/Desktop/val_name.txt'

with open(data_json) as f:
    f_data = f.read().decode('utf8')
print 'read file success...'

json_data = json.loads(f_data)
json_data = json_data['root']
print 'load json success...'

train_file = {}
val_file = {}
i = 0
for one in json_data:
    name = one['img_paths'].split('/')[-1]
    if name not in train_file and name not in val_file:
        if 'train' in name:
            train_file[name] = 1
        else:
            val_file[name] = 1
    else:
        if 'train' in name:
            train_file[name] += 1
        else:
            val_file[name] += 1
        #print name

print 'start write...'
train_data = []
for k, v in train_file.items():
    if v == 1:
        train_data.append(k)
train_data = '\n'.join(train_data)
with open(w_train_f_txt, 'w') as f:
    f.write(train_data)

val_data = []
for k, v in val_file.items():
    if v == 1:
        val_data.append(k)
val_data = '\n'.join(val_data)
with open(w_val_f_txt, 'w') as f:
    f.write(val_data)

print 'write success...'

