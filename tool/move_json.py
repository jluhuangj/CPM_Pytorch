import json
import sys
import os

data_json = '/Users/huangju/Desktop/COCO.json'
w_train_f_txt = '/Users/huangju/Desktop/train_name.json'
w_val_f_txt = '/Users/huangju/Desktop/val_name.json'

with open(data_json) as f:
    f_data = f.read().decode('utf8')
print 'read file success...'

json_data = json.loads(f_data)
json_data = json_data['root']
print 'load json success...'

train_file = []
val_file = []
for one in json_data:
    name = one['img_paths'].split('/')[-1]
    if name not in train_file and name not in val_file:
        if 'train' in name:
            train_file.append(name)
        else:
            val_file.append(name)
        print name

print 'start write...'

train_file = '\n'.join(train_file)
with open(w_train_f_txt, 'w') as f:
    f.write(train_file)

val_file = '\n'.join(val_file)
with open(w_val_f_txt, 'w') as f:
    f.write(val_file)

print 'write success...'

