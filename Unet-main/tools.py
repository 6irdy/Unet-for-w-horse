import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from Models import U_Net,NestedUNet    #插入
#from train import test

#from net import Unet_plus_plus
import torch.optim as optim
from torch.optim import lr_scheduler
from Dataset import HorseDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from computeloss import compute_iou, boundary_iou, bce_dice_loss, generate_boundary
from PIL import Image
import time

def test(model, test_data_loader):
    print("testing...")
    model.eval()
    IOU = []
    B_IOU = []
    LOSS = []

    with torch.no_grad():
        for test_data, test_mask in test_data_loader:

            loss = 0


            output = model(test_data)
            loss += bce_dice_loss(output, test_mask)
            iou = compute_iou(output.squeeze(dim=1), test_mask)
            b_iou = boundary_iou(test_mask, output.squeeze(dim=1))

            loss_num=loss.detach().numpy()
            IOU.append(iou)
            B_IOU.append(b_iou)
            LOSS.append(loss_num)
    mean_iou = sum(IOU) / len(IOU)
    mean_b_iou = sum(B_IOU) / len(B_IOU)
    mean_loss = sum(LOSS) / len(LOSS)
    print("测试集 mean iou", mean_iou)
    print("测试集 mean boundary_iou", mean_b_iou)
    print("测试集 mean loss", mean_loss)
    return mean_iou, mean_b_iou, mean_loss


def save_fig(plot_all,opt,TRAIN_IOU, TRAIN_LOSS, TEST_IOU, TEST_LOSS, LR, EPOCH, TRAIN_B_IOU, TEST_B_IOU):
    plt.figure(figsize=(8, 8))
    plt.title('Mean Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Iou')
    plt.plot(EPOCH, TRAIN_IOU, label='Train')
    plt.plot(EPOCH, TEST_IOU, label='Test')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_iou.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('Mean Boundary Iou')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Boundary Iou')
    plt.plot(EPOCH, TRAIN_B_IOU, label='Train')
    plt.plot(EPOCH, TEST_B_IOU, label='Test')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_TEST_boundary_iou.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('Mean Loss')
    plt.xlabel('EPOCH')
    plt.ylabel('Mean Loss')
    plt.plot(EPOCH, TRAIN_LOSS, label='Train')
    plt.plot(EPOCH, TEST_LOSS, label='Test')

    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/mean_loss.png" % (opt.outf))
    plt.clf()

    plt.figure(figsize=(8, 8))
    plt.title('learning rate')
    plt.xlabel('EPOCH')
    plt.ylabel('Learning rate')
    plt.plot(EPOCH, LR, label='lr')
    plt.legend(loc='upper left')
    if plot_all == 1:
        plt.show()
    else:
        plt.savefig(r"./%s/learning_rate.png" % (opt.outf))
    plt.clf()
    plt.close()


# 保存预测结果 包含语义分割预测结果，边界预测结果 和 实际图像分割标签和边界图像 以及 分割差值图像
def show_predict(opt,save_num,test_data_loader, sign):
    print("-" * 34)
    print("开始预测")
    # 两种载入一种依据模型参数
    best_model = U_Net()
    #best_model = Unet_plus_plus(deep_supervision=deep_supervision, cut=cut)  # 默认在CPU上
    state_dict = torch.load('best_model.pth')
    best_model.load_state_dict(state_dict, strict=False)



    if sign == 1:


        print("完整结构预测结果为:")
        test_mean_iou, test_mean_b_iou, test_mean_loss = test(best_model, test_data_loader)
    cal = 0
    toPIL = transforms.ToPILImage()  # 这个函数可以将张量转为PIL图片，由小数转为0-255之间的像素值
    with torch.no_grad():
        for test_data, test_mask in test_data_loader:
            cal += 1
            B, H, W = test_mask.shape
            for k in range(B):
                pic1 = toPIL(test_data[k])
                pic2 = toPIL(test_mask[k])

                pic1.save(r"./%s/%d_%d_ori.png" % (opt.predict, cal, k))
                pic2.save(r"./%s/%d_%d_mask.png" % (opt.predict, cal, k))

            outputs = best_model(test_data)
            # 获得边界
            mask_boundary = 255 * generate_boundary(test_mask, dilation_ratio=0.02, sign=1)
            '''
                        if deep_supervision is True:
                # print("采用深监督")
                out_boundary = 255 * mask_to_boundary(outputs[-1].squeeze(dim=1), dilation_ratio=0.02, sign=1)
                out = outputs[-1].squeeze(dim=1)
            else:
            '''

            # print("无深监督")
            out_boundary = 255 * generate_boundary(outputs.squeeze(dim=1), dilation_ratio=0.02, sign=1)
            out = outputs.squeeze(dim=1)#这里删去了squeeze

            Save_out = torch.sigmoid(out).data.cpu().numpy()
            Save_out[Save_out > 0.5] = 255
            Save_out[Save_out <= 0.5] = 0

            test_mask_ = torch.sigmoid(test_mask).data.cpu().numpy()
            test_mask_[test_mask_ > 0.5] = 255
            test_mask_[test_mask_ <= 0.5] = 0
            for j in range(B):
                A = Image.fromarray(mask_boundary[j].astype('uint8'))
                B = Image.fromarray(out_boundary[j].astype('uint8'))
                A.save(r"./%s/%d_%d_mask_boundary.png" % (opt.predict, cal, j))
                B.save(r"./%s/%d_%d_predict_boundary.png" % (opt.predict, cal, j))
                Y = test_mask_[j].astype('uint8')
                X = Save_out[j].astype('uint8')
                sub = np.abs(X - Y)
                sub = Image.fromarray(sub)
                sub.save(r"./%s/%d_%d_sub.png" % (opt.predict, cal, j))
                Z = Image.fromarray(X)
                Z.save(r"./%s/%d_%d_predict.png" % (opt.predict, cal, j))
            if (cal == save_num):
                print("预测结束")
                print("-" * 34)
                break




