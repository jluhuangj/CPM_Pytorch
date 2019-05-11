"""


"""
import os
import time
import torch
import torch.optim as optim
import torch.nn as nn
import ConfigParser

from torch.autograd import Variable
from torch.utils.data import DataLoader

from data_loader.coco_pose_data import COCOPoseDataset as Mydata
from model.cpm import CPM
from src.util import *

# multi-GPU
device_ids = [0, 1, 2, 3]

# *********************** hyper parameter  ***********************

config = ConfigParser.ConfigParser()
config.read('conf.text')
train_data_dir = config.get('data', 'train_data_dir')
train_label_dir = config.get('data', 'train_label_dir')
save_dir = config.get('data', 'save_dir')

learning_rate = config.getfloat('training', 'learning_rate')
batch_size = config.getint('training', 'batch_size')
epochs = config.getint('training', 'epochs')
begin_epoch = config.getint('training', 'begin_epoch')

cuda = torch.cuda.is_available()

if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# *********************** Build dataset ***********************
train_data = Mydata(data_dir=train_data_dir, label_dir=train_label_dir)
print 'Train dataset total number of images sequence is ----' + str(len(train_data))

# Data Loader
train_dataset = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# *********************** Build model ***********************

net = CPM(out_c=18)

if cuda:
    net = net.cuda()


if begin_epoch > 0:
    save_path = 'ckpt/model_epoch' + str(begin_epoch) + '.pth'
    state_dict = torch.load(save_path)
    net.load_state_dict(state_dict)


def train():
    # *********************** initialize optimizer ***********************
    optimizer = optim.Adam(params=net.parameters(), lr=learning_rate, betas=(0.5, 0.999))
    criterion = nn.MSELoss(size_average=True)                       # loss function MSE average

    net.train()
    for epoch in range(begin_epoch+1, epochs + 1):
        epoch_start_time = time.time()
        c_time = time.time()
        for step, (image, label_map, imgs) in enumerate(train_dataset):
            image = Variable(image.cuda() if cuda else image)                   # 4D Tensor
            # Batch_size  *  3  *  width(368)  *  height(368)

            # 4D Tensor to 5D Tensor
            label_map = torch.stack([label_map]*6, dim=1)
            # Batch_size  *  21 *   45  *  45
            # Batch_size  *   6 *   21  *  45  *  45
            label_map = Variable(label_map.cuda() if cuda else label_map)

            optimizer.zero_grad()
            pred_6 = net(image)  # 5D tensor:  batch size * stages * 21 * 45 * 45

            # ******************** calculate loss of each joints ********************
            loss = criterion(pred_6, label_map)

            # backward
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                print 'Train Epoch: {0} [{1}/{2} ({3:.0f}%)], Took: {4:d}h {5:d}m {6:.2f}s, Loss:{7:.6f},'.format(epoch,
                        step*(batch_size), len(train_dataset.dataset),
                        100.*step*(batch_size) / len(train_dataset.dataset),
                        int(time.time() - c_time)/3600,
                        int(time.time() - c_time)%3600/60,
                        int(time.time() - c_time)%3600%60,
                        loss.item())
                c_time = time.time()

            if step % 200 == 0:
                save_images(label_map[:, 5, :, :, :], pred_6[:, 5, :, :, :], step, epoch, imgs)

        torch.save(net.state_dict(), os.path.join(save_dir, 'model_epoch{:d}.pth'.format(epoch)))

        print 'current time: {0} epoch {1} spend {2:d}h {3:d}m {4:d}s ...'.format(
                time.asctime(time.localtime(time.time())),
                step,
                int(time.time() - epoch_start_time)/3600,
                int(time.time() - epoch_start_time)%3600/60,
                int(time.time() - epoch_start_time)%3600%60)

    print 'train done!'


if __name__ == '__main__':
    train()

