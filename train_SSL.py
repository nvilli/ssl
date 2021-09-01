# 2021-09-01 this file is built to train model by self-supervised trainging strategy

from config import ActivityConfig as cfg
from tools import accuracy, AverageMeter, print_config, parse_args, info_one_step, info_one_epoch, save_config
from torch import nn, optim
import os
from models import C3D
from datasets import ucf101
import time
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
import random
import numpy as np
from apex import amp
from prefetch_generator import BackgroundGenerator


def train(train_loader, model, criterion_CE, optimizer, epoch, root_path=None):
    torch.set_grad_enabled(True)

    batch_time = AverageMeter()                                 # how much time that one batch spend
    data_time = AverageMeter()                                  # how much time that one batch data loaded spend
    losses_CE = AverageMeter()                                  # calculate cross-entropy loss
    losses = AverageMeter()                                     # calculate total loss
    acc = AverageMeter()                                        # calculate prediction accuracy
    end = time.time()                                           # when one epoch training loop end
    model.train()

    # modify here
    total_cls_loss = 0.0                                    # total classification loss
    correct_cnt = 0                                         # correct classification result
    total_cls_cnt = torch.zeros(cfg.DATASET.CLASS_NUM)      # total correct classification result
    correct_cls_cnt = torch.zeros(cfg.DATASET.CLASS_NUM)    # count how many class be classified correctly

    # training loop
    # return clip_rgb, clip_diff, sample_step_label, p_label
    if cfg.TRAIN.TYPE == 'SSL':
        for step, (clip_rgb, clip_diff, sample_step_label, p_label) in enumerate(train_loader):
            data_time.update(time.time() - end)

            # prepare data and send data to CUDA
            clip_rgb = clip_rgb.cuda()
            clip_diff = clip_diff.cuda()
            sample_step_label = sample_step_label.cuda()
            p_label = p_label.cuda()
            input_tensor = torch.cat((clip_rgb, clip_diff), dim=2)
            input_tensor = input_tensor.cuda()

            # calculate result and loss
            rgb_output, diff_output = model(input_tensor)
            final_result = None
            if cfg.TRAIN.FUSION == 'weight-average':
                final_result = rgb_output * cfg.TRAIN.RGB_WEIGHT + diff_output * cfg.TRAIN.DIFF_WEIGHT
            loss_class = criterion_CE(final_result, p_label)
            loss = loss_class

            # update model
            optimizer.zero_grad()
            # use APEX to accelerate not only the loss.backward(), but the whole training loop
            if cfg.APEX:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif not cfg.APEX:
                loss.backward()
            optimizer.step()

            # update information
            losses_CE.update(loss_class.item(), input_tensor.size(0))
            losses.update(loss.item(), input_tensor.size(0))
            batch_time.update(time.time() - end)
            end = time.time()
            prec_class = accuracy(final_result.data, p_label, topk=(1,))[0]
            acc.update(prec_class.item(), input_tensor.size(0))

            # print and save info at every 20 steps
            if (step + 1) % cfg.SHOW_INFO == 0:
                total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt = info_one_step(root_path, batch_time,
                                                                                            data_time, losses,
                                                                                            acc, final_result,
                                                                                            p_label, losses_CE,
                                                                                            step, epoch,
                                                                                            len(train_loader),
                                                                                            [optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']],
                                                                                            total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt)
        # print and save info at one epoch
        info_one_epoch(root_path, total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt, len(train_loader), log_type='train')



def validation(valid_loader, model, criterion_CE, optimizer, epoch, root_path=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses_CE = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    model.eval()
    end = time.time()
    total_loss = 0.0

    # modify here
    total_cls_loss = 0.0  # total classification loss
    correct_cnt = 0  # correct classification result
    total_cls_cnt = torch.zeros(cfg.DATASET.CLASS_NUM)  # total correct classification result
    correct_cls_cnt = torch.zeros(cfg.DATASET.CLASS_NUM)  # count how many class be classified correctly

    # validation loop
    # return clip_rgb, clip_diff, sample_step_label, p_label
    with torch.no_grad():
        if cfg.TRAIN.TYPE == 'SSL':
            for step, (clip_rgb, clip_diff, sample_step_label, p_label) in enumerate(valid_loader):
                data_time.update(time.time() - end)

                # prepare data and send data to CUDA
                clip_rgb = clip_rgb.cuda()
                clip_diff = clip_diff.cuda()
                sample_step_label = sample_step_label.cuda()
                p_label = p_label.cuda()
                input_tensor = torch.cat((clip_rgb, clip_diff), dim=2)
                input_tensor = input_tensor.cuda()

                # calculate result and loss
                rgb_output, diff_output = model(input_tensor)
                final_result = None
                if cfg.TRAIN.FUSION == 'weight-average':
                    final_result = rgb_output * cfg.TRAIN.RGB_WEIGHT + diff_output * cfg.TRAIN.DIFF_WEIGHT
                loss_class = criterion_CE(final_result, p_label)
                loss = loss_class

                # update information
                losses_CE.update(loss_class.item(), input_tensor.size(0))
                losses.update(loss.item(), input_tensor.size(0))
                batch_time.update(time.time() - end)
                end = time.time()
                total_loss += loss.item()
                prec_class = accuracy(final_result, p_label, topk=(1,))[0]
                acc.update(prec_class.item(), input_tensor.size(0))

                # print and save information at every 20 steps
                if (step + 1) % cfg.SHOW_INFO == 0:
                    total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt = info_one_step(root_path, batch_time,
                                                                                                data_time, losses, acc,
                                                                                                final_result, p_label,
                                                                                                losses_CE, step, epoch,
                                                                                                len(valid_loader),
                                                                                                [optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']],
                                                                                                total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt)
    # print and save info at one epoch
    info_one_epoch(root_path, total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt, len(valid_loader), log_type='val')

    avg_loss = losses.avg
    return avg_loss


def main():
    # prepare environment, and show config information
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    cfg.GPUS = args.gpus
    print_config()

    # set path to save model and log, and save config parameter as a log file
    save_path = os.path.join(cfg.SAVE_PATH, cfg.EXP_TAG, cfg.TRAIN.TYPE)
    if not os.path.exists(save_path):
        print("===> Workspace path is not built, will be created!")
        os.makedirs(save_path)
        print("===> Workspace has been built!")
    save_config(save_path)

    # get model
    model = None
    if cfg.MODEL_NAME == 'c3d':
        model = C3D.C3D(num_classes=cfg.DATASET.CLASS_NUM, train_type=cfg.TRAIN.TYPE)

    # get dataset, and build dataloader for training, validating and testing
    train_dataset = None
    valid_dataset = None
    if cfg.DATASET.NAME == "UCF-101-origin":
        dataset = ucf101.SSL_Dataset(root=os.path.join(cfg.DATASET.ROOT_PATH, cfg.DATASET.NAME), mode='train', args=args)
        val_size = cfg.DATASET.VAL_SIZE
        train_dataset, valid_dataset = random_split(dataset, (len(dataset) - val_size, val_size))
    # build data loader, using DataLoaderX or not
    train_loader = None
    valid_loader = None
    if cfg.DATASET.LOAD_TYPE == 'normal':
        train_loader = DataLoader(train_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  drop_last=True)
        valid_loader = DataLoader(valid_dataset,
                                  batch_size=cfg.TRAIN.BATCH_SIZE,
                                  shuffle=True,
                                  num_workers=cfg.TRAIN.NUM_WORKERS,
                                  drop_last=True)
    elif cfg.DATASET.LOAD_TYPE == 'DataLoaderX':
        train_loader = ucf101.DataLoaderX(train_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.TRAIN.NUM_WORKERS,
                                          drop_last=True)
        valid_loader = ucf101.DataLoaderX(valid_dataset,
                                          batch_size=cfg.TRAIN.BATCH_SIZE,
                                          shuffle=True,
                                          num_workers=cfg.TRAIN.NUM_WORKERS,
                                          drop_last=True)

    # prepare other components, send data and model into CUDA
    if cfg.MULTI_GPU:
        model = nn.DataParallel(model)
    model = model.cuda()
    criterion_CE = nn.CrossEntropyLoss().cuda()

    # set optimizer and scheduler
    model_params = []
    for key, value in dict(model.named_parameters()).items():
        if value.requires_grad:
            model_params += [{'params': [value], 'lr':cfg.TRAIN.LEARNING_RATE}]
    optimizer = optim.SGD(model_params,
                          momentum=cfg.TRAIN.MOMENTUM,
                          weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                     'min',
                                                     min_lr=cfg.TRAIN.MIN_LR,
                                                     patience=cfg.TRAIN.PATIENCE,
                                                     factor=cfg.TRAIN.FACTOR)

    # using APEX to accelerate training
    if cfg.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # train and validate loop
    prev_best_val_loss = 100
    prev_best_loss_model_path = None
    for epoch in tqdm(range(cfg.TRAIN.START_EPOCH, cfg.TRAIN.START_EPOCH+cfg.TRAIN.EPOCH)):
        # train model
        # def train(train_loader, model, criterion_CE, optimizer, epoch, root_path=None):
        train(train_loader, model, criterion_CE, optimizer, epoch, root_path=save_path)
        # validate model
        # def validation(valid_loader, model, criterion_CE, optimizer, epoch, root_path):
        val_loss = validation(valid_loader, model, criterion_CE, optimizer, epoch, root_path=save_path)
        # save model if current model is better than previous
        if val_loss < prev_best_val_loss:
            model_path = os.path.join(save_path, 'best_val_loss_model_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), model_path)
            prev_best_val_loss = val_loss
            if prev_best_loss_model_path:
                os.remove(prev_best_loss_model_path)
            prev_best_loss_model_path = model_path
        scheduler.step(val_loss)

        # save checkpoints
        if epoch % cfg.SHOW_INFO == 0:
            checkpoints = os.path.join(save_path, 'model_checkpoint_{}.pth.tar'.format(epoch))
            torch.save(model.state_dict(), checkpoints)
            print('===> Checkpoint will be saved to: ', checkpoints)


if __name__ == '__main__':
    seed = cfg.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    main()