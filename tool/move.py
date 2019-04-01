import os
import sys
import shutil

src_dir = '/Users/huangju/Desktop/train2014'
dst_dir = '/Users/huangju/Desktop/train2014_pose'
conf = '/Users/huangju/Desktop/train_name.txt'

if not os.path.exists(dst_dir):
    os.mkdir(dst_dir)

print 'start reading conf...'
with open(conf) as f:
    f_data = f.readlines()
f_data = [line.strip().decode('utf8') for line in f_data]

print 'start moveing ...'
for name in f_data:
    print name
    src_path = os.path.join(src_dir, name)
    dst_path = os.path.join(dst_dir, name)
    try:
        shutil.move(src_path, dst_path)
    except IOError:
        print 'no {}'.format(name)

print 'success...'



