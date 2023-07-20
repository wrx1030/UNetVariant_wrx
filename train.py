import argparse
import logging
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.backends import cudnn
from tqdm import tqdm

from evaluate import evaluate
from model import newUNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split

# dir_img = 'D:/DataSet/Company/company_sj/TrainingData/images/'
# dir_mask = 'D:/DataSet/Company/company_sj/TrainingData/masks/'
# dir_previous = 'D:/DataSet/Company/company_sj/TrainingData/previous/'
# dir_img = 'D:/DataSet/LiTs/2.0_ctpng/'
# dir_mask = 'D:/DataSet/LiTs/2.0_labelpng/'
# dir_previous = 'D:/DataSet/LiTs/2.0_prect/'
dir_img = 'D:/DataSet/3Dircadb/next_ct/trainct/'
dir_mask = 'D:/DataSet/3Dircadb/next_ct/traingt/'
dir_previous = 'D:/DataSet/3Dircadb/next_ct/trainnextct/'
dir_checkpoint = 'checkpoints/'

def train_net(net,
              device,
              epochs,
              batch_size,
              lr,
              val_percent,
              img_scale,
              save_cp=True
              ):

    dataset = BasicDataset(dir_img, dir_mask, dir_previous, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')

    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9) #优化器 RMSprop算法
    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)  # 优化器 Adam算法
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3) #调整学习率
    seg_criterion = nn.BCEWithLogitsLoss() #loss
    previous_criterion = nn.MSELoss()
    step_num = n_train//batch_size

    Resume_train = True  # 是否继续训练 True False
    if Resume_train:
        path_checkpoint = "./checkpoints/3Dircadb_groupnorm_epoch10.pth"  # 断点路径
        checkpoint = torch.load(path_checkpoint)  # 加载断点
        net.load_state_dict(checkpoint['net'])  # 加载模型可学习参数
        optimizer.load_state_dict(checkpoint['optimizer'])  # 加载优化器参数
        start_epoch = checkpoint['epoch']  # 设置开始的epoch

    # for epoch in range(epochs):  # 初次训练
    for epoch in range(start_epoch + 1, start_epoch + epochs + 1):  # 继续训练
        net.train()
        epoch_loss = 0
        # 初次训练进度条
        # with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        # 续训练进度条
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{start_epoch + epochs + 1}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                previous = batch['previous']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                previous = previous.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.n_classes == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred, previous_pred = net(imgs)
                loss1 = seg_criterion(masks_pred, true_masks)
                loss2 = previous_criterion(previous_pred, previous)
                epoch_loss += loss1.item()
                writer.add_scalar('Loss/train', loss1.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss1.item()})

                optimizer.zero_grad()
                loss2.backward(retain_graph=True)
                loss1.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1) #梯度裁剪
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                # 验证集调整学习率

        epoch_metric = epoch_loss/step_num
        scheduler.step(epoch_metric)
        print(epoch_metric, optimizer.state_dict()['param_groups'][0]['lr'])
        # scheduler.best 保存着当前模型中的指标最小模型

        if save_cp and (epoch+1) % 10 == 0:  # 保存节点频率调整
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            checkpoint = {
                "net": net.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch
            }
            torch.save(checkpoint,
                       dir_checkpoint + f'3Dircadb_groupnorm_epoch{epoch + 1}.pth')  # 记得改名
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = newUNet(n_channels=1, n_classes=1)
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Transposed conv"} upscaling')

    net.to(device=device)
    # faster convolutions, but more memory
    cudnn.benchmark = True

    try:
        train_net(net=net, device=device, epochs=90, batch_size=3, lr=1e-4, val_percent=0, img_scale=1)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)

# if global_step % (n_train // (200 * batch_size)) == 0:
                #     # for tag, value in net.named_parameters():
                #     #     tag = tag.replace('.', '/')
                #     #     writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                #     #     writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                #     val_score = evaluate(net, val_loader, device)
                #     scheduler.step(val_score)
                #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
                #
                #     if net.n_classes > 1:
                #         logging.info('Validation cross entropy: {}'.format(val_score))
                #         writer.add_scalar('Loss/test', val_score, global_step)
                #     else:
                #         logging.info('Validation Dice Coeff: {}'.format(val_score))
                #         writer.add_scalar('Dice/test', val_score, global_step)
                #
                #     writer.add_images('images', imgs, global_step)
                #     if net.n_classes == 1:
                #         writer.add_images('masks/true', true_masks, global_step)
                #         writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
