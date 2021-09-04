# 2021-09-01 this file is built to train model by self-supervised trainging strategy

# from enum import Flag
from apex.amp.utils import is_fp_tensor
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
import sys


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
                                                                                            p_label, loss_class,
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
                                                                                                loss_class, step, epoch,
                                                                                                len(valid_loader),
                                                                                                [optimizer.param_groups[0]['lr'], optimizer.param_groups[-1]['lr']],
                                                                                                total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt)
    # print and save info at one epoch
    info_one_epoch(root_path, total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt, len(valid_loader), log_type='val')

    avg_loss = losses.avg
    return avg_loss


def main(id_str, cnt):
    # prepare environment, and show config information
    args = parse_args()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    os.environ["CUDA_VISIBLE_DEVICES"] = id_str
    cfg.GPUS = cnt
    cfg.GPU_ID = id_str
    # cfg.GPUS = args.gpus
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

    # load model to CUDA
    model = model.cuda()

    # using APEX to accelerate training
    if cfg.APEX:
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")

    # prepare other components, send data and model into CUDA
    if cfg.MULTI_GPU:
        model = nn.DataParallel(model)
    
    criterion_CE = nn.CrossEntropyLoss().cuda()

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

def gpu_info():
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')
    gpu_memory_info = []
    for i in range(8):
        gpu_memory_info.append(gpu_status[i * 4 + 2])
    return gpu_memory_info

def info_parse(gpu_memory_info):
    gpu_memory_free = []
    for i in range(len(gpu_memory_info)):
        used_memory  = next(i for i, j in enumerate(gpu_memory_info[i].split('/')[0].split(' ')) if j)
        total_memory = next(i for i, j in enumerate(gpu_memory_info[i].split('/')[1].split(' ')) if j)
        used_memory  = gpu_memory_info[i].split('/')[0].split(' ')[used_memory].split('MiB')[0]
        total_memory = gpu_memory_info[i].split('/')[1].split(' ')[total_memory].split('MiB')[0]
        gpu_memory_free.append(int(total_memory) - int(used_memory))
    
    return gpu_memory_free

def check_info(gpu_need, gpu_memory_free):

    info_dict = {}
    for i in range(len(gpu_memory_free)):
        info_dict.update({str(gpu_memory_free[i]): i})

    gpu_memory_free.sort(reverse=True)
    room_free = 0
    gpu_temp = []
    gpu_cnt = 0
    USE_FLAG = False
    # check whether there is enough room for training
    for i in range(len(gpu_memory_free)):
        # check whether there is at least one gpu to use
        if room_free < gpu_need:
            if gpu_cnt < 8:
                room_free += int(gpu_memory_free[i])
                gpu_temp.append(gpu_memory_free[i])
                gpu_cnt += 1
            elif gpu_cnt > 8:
                break
        # check whether requirement fulfilled
        if room_free >= gpu_need:
            final_flag = len(gpu_temp) - 1
            if gpu_temp[final_flag] * gpu_cnt >= gpu_need:
                USE_FLAG = True
                break
            elif gpu_temp[final_flag] + gpu_cnt < gpu_need:
                USE_FLAG = False

    _id = []
    for i in range(len(gpu_temp)):
        _id.append(info_dict[str(gpu_temp[i])])
    
    return room_free, gpu_temp, gpu_cnt, USE_FLAG, _id

def _watch_gpu(interval = 0.1):
    t = 0
    FLAG = True
    _totalfree = 0
    _freeid = None
    cnt = 0

    # argument parse
    args = parse_args()
    gpu_need = args.gpus

    while True and FLAG:
        gpu_memory_info = gpu_info()
        gpu_memory_free = info_parse(gpu_memory_info)
        room_free, gpu_temp, gpucnt, useflag, _id = check_info(gpu_need, gpu_memory_free)
        
        t = t % 5
        symbol = '===> Monitoring: ' + '>' * t + ' ' * (5 - t - 1) + ' | '
        id_str = 'GPU id: ' + str(_id) + ' | '
        room_str = 'Memory info: ' + str(gpu_temp) + ' | '
        time_str = 'Time: ' + time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
        sys.stdout.write('\r' + symbol + id_str + room_str + time_str)
        sys.stdout.flush()
        time.sleep(interval)
        t += 1

        if useflag == True:
            _totalfree = room_free
            _freeid = _id
            cnt = gpucnt
            FLAG = False

    return _totalfree, _freeid, cnt

if __name__ == '__main__':
    seed = cfg.RANDOM_SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


    print('-------------------------------------------------------------------------------')
    print('===> Start time:{}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))
    totalfree, freeid, cnt = _watch_gpu()
    print("===> Exp will start at: ", time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    print("===> GPU info: Free memory-", str(totalfree), " GPU id-", str(freeid), " Count-", str(cnt))
    print('-------------------------------------------------------------------------------')

    id_str = ''
    freeid.sort()
    for i in range(len(freeid)):
        if i == 0:
            id_str += str(freeid[i])
        elif i != 0:
            sub_str = ',' + str(freeid[i])
            id_str += sub_str
    main(id_str, cnt)