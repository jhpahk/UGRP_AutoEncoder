from enum import auto
import torch
from torch import optim
import torch.nn as nn

import time
import os

from encoder import AutoEncoder
from dataset import COCO_val
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


now = time.localtime()

time_path = 'checkpoint_%04d_%02d_%02d_%02d:%02d:%02d' % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min, now.tm_sec)
checkpoint_dir_path = "./checkpoint/" + time_path
os.makedirs(checkpoint_dir_path)
checkpoint_file_path = time_path + ".pth"
CHECKPOINT_PATH = os.path.join(checkpoint_dir_path, checkpoint_file_path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
writer = SummaryWriter()


coco_data = COCO_val()
train_loader = DataLoader(coco_data, batch_size=4, shuffle=True)

autoencoder = AutoEncoder().cuda()
autoencoder = nn.DataParallel(autoencoder)

learning_rate = 1e-5
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=learning_rate)

epochs = 1

for epoch in range(epochs):
    for data in train_loader:
        data = data.cuda()
        out = autoencoder(data)
        loss = loss_fn(data, out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"loss: {loss}")

    if epoch % 10 == 0 and epoch != 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, CHECKPOINT_PATH)

    writer.add_scalar("Train Loss", loss, epoch)