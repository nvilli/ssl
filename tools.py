from config import ActivityConfig as cfg
import argparse
import torch
import os
import sys


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def print_config():
    # print(cfg)
    for key, value in cfg.items():
        print(str(key) + ": " + str(value))


def save_config(root_path):
    conf_log_path = os.path.join(root_path, "conf.log")
    conf_log = open(conf_log_path, "a")
    for key, value in cfg.items():
        s_str = str(key) + ": " + str(value) + '\n'
        conf_log.write(s_str)
    conf_log.close()
    print("===> Config log save success!")


def parse_args():
    parser = argparse.ArgumentParser(description='SSL_learning')
    parser.add_argument('--gpu_id', type=str, default='0', help='GPU id')
    parser.add_argument('--gpus', type=int, default=1, help='gpus that can be used')
    args = parser.parse_args()
    return args


def info_one_step(save_path, batch_time, data_time,
                  losses, acc, result, label, loss_class,
                  step, epoch, loader_len, optimizer_para,
                  total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt):

    log_file_name = cfg.TIMESTAMP + '.log'
    log_file_path = os.path.join(save_path, log_file_name)
    file = open(log_file_path, 'a')

    total_cls_loss += loss_class.item()
    pts = torch.argmax(result, dim=1)
    correct_cnt += torch.sum(label == pts).item()
    for i in range(label.size(0)):
        total_cls_cnt[label[i]] += 1
        if label[i] == pts[i]:
            correct_cls_cnt[pts[i]] += 1

    # if (step + 1) % cfg.SHOW_INFO == 0:
    p_str = "-------------------------------------------------------------------"
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> Epoch:[{0}][{1}/{2}]".format(epoch, step + 1, loader_len)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> Conv lr: {} FC lr: {}".format(optimizer_para[0], optimizer_para[1])
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> Data time:{data_time:.3f}  Batch time:{batch_time:.3f}".format(data_time=data_time.val,
                                                                                      batch_time=batch_time.val)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> Loss:{loss:.5f}".format(loss=losses.avg)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    p_str = "===> Accuracy:{acc:.3f}".format(acc=acc.avg)
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(str(s_str))

    file.close()

    return total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt


def info_one_epoch(save_path, total_cls_loss, correct_cnt, total_cls_cnt, correct_cls_cnt, loader_len, log_type='train'):

    log_file_name = cfg.TIMESTAMP + '.log'
    log_file_path = os.path.join(save_path, log_file_name)
    file = open(log_file_path, 'a')

    avg_cls_loss = total_cls_loss / loader_len
    avg_acc = correct_cnt / loader_len

    if log_type == 'train':
        p_str = '===> [TRAIN] Loss: {:.3f}  Acc: {:.3f}'.format(avg_cls_loss, avg_acc)
        print(p_str)
        s_str = str(p_str) + '\n'
        file.write(s_str)
    if log_type == 'val':
        p_str = '===> [VAL] Loss: {:.3f}  Acc: {:.3f}'.format(avg_cls_loss, avg_acc)
        print(p_str)
        s_str = str(p_str) + '\n'
        file.write(s_str)

    p_str = '===> Correct classify count: ', correct_cls_cnt
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(s_str)

    p_str = '===> Total classify count: ', total_cls_cnt
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(s_str)

    p_str = '===> Total classify acc: ', correct_cls_cnt / total_cls_cnt
    print(p_str)
    s_str = str(p_str) + '\n'
    file.write(s_str)

    file.close()
