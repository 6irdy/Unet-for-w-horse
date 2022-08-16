'''
Unet implemented on pytorch by 6irdy
train Unet on Weizmann horse dataset
需要将数据集按照指定位置放置，这一版本上传的是Unet训练版本，子目录下的权重均为Unet使用
如需使用NesTEDUnet参数，请调用epoch200中的best_model.pth以及setting.txt进行训练或者预测。
'''
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from Models import U_Net,NestedUNet
from tools import save_fig,show_predict,test

import torch.optim as optim
from torch.optim import lr_scheduler
from Dataset import HorseDataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from computeloss import compute_iou, boundary_iou, bce_dice_loss, generate_boundary
from PIL import Image
import time



#---------

# 训练过程
def train(model, data_loader, optimizer):
    print("training...")
    model.train()
    # 记录iou,boundary iou,loss
    IOU = []
    B_IOU = []
    LOSS = []
    num = 0
    # 一整个Epoch训练过程
    for image_batch, mask_batch in data_loader:
        num += 1
        loss = 0

        output = model(image_batch)
        loss = bce_dice_loss(output, mask_batch)

        iou = compute_iou(output.squeeze(dim=1), mask_batch)
        b_iou = boundary_iou(mask_batch, output.squeeze(dim=1))
        loss_num = loss.detach().numpy()
        print(num, "iou:", iou, "boundary_iou:", b_iou, "loss:", loss_num)
        # 记录数据

        LOSS.append(loss_num)
        IOU.append(iou)
        B_IOU.append(b_iou)

        # 参数更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    mean_iou = sum(IOU) / len(IOU)
    mean_b_iou = sum(B_IOU) / len(B_IOU)
    mean_loss = sum(LOSS) / len(LOSS)
    print("训练集 mean iou", mean_iou)
    print("训练集 mean boundary_iou", mean_b_iou)
    print("训练集 mean loss", mean_loss)
    return mean_iou, mean_b_iou, mean_loss


def parse_opt(known=False):
    parser = argparse.ArgumentParser()
    # ---------------------------------------------------------------------------
    parser.add_argument('--root', default='D:/Dataset/horse/weizmann_horse_db', help='folder to Dataset')# 指向数据集储存位置
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train for') # epoch次数
    parser.add_argument('--new_model', type=int, help='new model whether or not', default=0) # 是否新建模型    1-新建    0-使用历史最佳pth
    parser.add_argument('--lr', type=float, default=1e-4, help='the learning rate')#初始学习率
    parser.add_argument('--min_lr', type=float, default=1e-6, help='minimum learning rate')# 最小学习率
    parser.add_argument('--early_stop_b_iou', type=float, default=0.69, help='the minimal boundary iou')# 保存模型条件    达到此要求会保存一个模型参数 best_biou.pth，由于我们开始训练时并不能达到0.69和0.9的水平，我们将会使用0.35和0.8进行替代。
    parser.add_argument('--early_stop_iou', type=float, default=0.90, help='the minimal iou')# 保存模型条件    达到此要求会保存一个模型参数 best_iou.pth
    parser.add_argument('--signal_iou', type=float, default=0.90, help='the signal iou')# 退出训练条件之一 达到此要求代表有机会可以停止训练
    # ---------------------------------------------------------------------------
    parser.add_argument('--shuffle', type=bool, default=False, help='shuffle or not')# 是否打乱数据集
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=0)# 使用多少个子进程导入数据,0代表无子进程，1代表共两个进程，以此类推
    parser.add_argument('--cuda', action='store_true', help='enables cuda')# 是否使用gpu训练
    parser.add_argument('--basic_channel', type=int, default=32, help='Batch size') # 网络基本通道数
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')# batch大小
    parser.add_argument('--total_num', type=int, default=327, help='Number of total images')# 数据集个数
    parser.add_argument('--train_size', type=int, default=278, help='Number of training images')# 训练集个数
    parser.add_argument('--test_size', type=int, default=49, help='Number of test images')# 测试集样本个数
    parser.add_argument('--image_size', type=int, default=80, help='the height / width of the input image to network')# image_size
    parser.add_argument('--save_num', type=int, default=49, help='Number of predict images')# 保存预测图片的batch数
    parser.add_argument('--nesterov', type=bool, default=True, help='the momentum')# SGD参数
    parser.add_argument('--momentum', type=float, default=0.9, help='the momentum')# 学习率衰减 及 参数控制 采用SGD时才有效
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')# 学习率衰减 及 参数控制
    parser.add_argument('--early_stopping', default=10, type=int, help='early stopping')# 早停标准
    parser.add_argument('--plot_all', type=int, default=0, help='1 == plot all images')# plot_all表示是否显示图片
    parser.add_argument('--outf', default='./logs', help='folder to train_log')# 输出训练结果输出到目标目录
    parser.add_argument('--predict', default='./predict', help='folder to predict')# 输出训练预测结果输出到目标目录
    parser.add_argument('--manualSeed', type=int, default=11, help='manual seed')# 设定随机种子

    opt = parser.parse_known_args()[0] if known else parser.parse_args()
    return opt

def main(opt):
    #

    argsDict = opt.__dict__
    # 写入参数设置至setting.txt
    with open('setting.txt', 'w') as f:
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- end -------------------')

    # 构造输出目录
    try:
        os.makedirs(opt.outf)
        os.makedirs(opt.predict)
    except OSError:
        print("already exist")

    # 设置随机种子
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    # 载入种子
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    random.seed(opt.manualSeed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 载入参数
    # 训练过程中参数赋值
    workers = opt.workers
    new_model = opt.new_model
    shuffle = opt.shuffle
    img_path = opt.root
    save_num = opt.save_num
    epoch_num = opt.epochs
    batch_size = opt.batch_size
    total_num = opt.total_num
    early_stop_b_iou = opt.early_stop_b_iou
    early_stop_iou = opt.early_stop_iou
    signal_iou = opt.signal_iou
    train_size = opt.train_size
    test_size = opt.test_size
    image_size = opt.image_size
    learning_rate = opt.lr
    min_learning_rate = opt.min_lr
    plot_all = opt.plot_all
    weight_decay = opt.weight_decay
    early_stopping = opt.early_stopping

    # 训练集加测试集不能超过样本总数
    assert train_size + test_size <= total_num, "Traing set size + Test set size > Total dataset size"

    # 划分训练集and 测试集
    idx = np.arange(total_num)
    if shuffle:
        np.random.shuffle(idx)
        print("数据打乱")
    else:
        print("数据未打乱")
    training_idx = idx[:train_size]
    testing_idx = idx[train_size:train_size + test_size]

    # 图像数据预处理  and 数据增强
    train_transforms = Transform_Compose([Train_Transform(image_size=image_size), Totensor()])
    test_transforms = Transform_Compose([Test_Transform(image_size=image_size), Totensor()])

    # 载入数据
    print("-" * 30)
    print("载入数据...")
    # 分别载入训练数据和测试数据
    Train_data = HorseDataset(img_path, training_idx, train_transforms)
    Test_data = HorseDataset(img_path, testing_idx, test_transforms)
    train_data_loader = torch.utils.data.DataLoader(
        Train_data,
        batch_size=batch_size,
        num_workers=workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True
    )

    # 为了能够复现模型结果，将测试集的batch_size设置1,并不打乱数据集
    test_data_loader = torch.utils.data.DataLoader(
        Test_data,
        batch_size=1,
        num_workers=workers,
        shuffle=False,
        pin_memory=True,
        drop_last=True
    )
    #///////////////////
    print("载入Unet模型...")
    if new_model == 1:
        print("新建模型")
        #model =NestedUNet()
        model = U_Net()


    elif new_model == 0:
        print("载入历史最佳模型")
        model=U_Net()
        #model = Unet_plus_plus(input_channel=3, num_classes=1, deep_supervision=deep_supervision, cut=False)
        state_dict = torch.load('Unetparam/best_model.pth')  # 载入参数
        model.load_state_dict(state_dict, strict=False)
    else:
        model=U_Net()
        #model = Unet_plus_plus(input_channel=3, num_classes=1, deep_supervision=deep_supervision, cut=False)

    # 定义需要更新的参数
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]

    # 定义优化器 经过实验Adam效果比较好
    print("定义优化器...")
    Optimizer = optim.Adam(params_to_optimize, lr=learning_rate, weight_decay=weight_decay)
    # Optimizer = optim.SGD(params_to_optimize, lr=learning_rate, momentum=momentum,nesterov=nesterov, weight_decay=weight_decay)

    # 定义学习率规划
    print("定义学习率规划...")
    # 采用模拟余弦退火规划学习率
    my_lr_scheduler = lr_scheduler.CosineAnnealingLR(Optimizer, T_max=epoch_num, eta_min=min_learning_rate)

    break_sign = 0  # 退出信号
    best_iou = early_stop_iou  # 用于记录最佳iou
    best_boundary_iou = early_stop_b_iou  # 用于记录最佳boundary iou

    #存储IOU BIOU LOSS
    TRAIN_ALL_IOU = []
    TRAIN_ALL_B_IOU = []
    TRAIN_ALL_LOSS = []
    TEST_ALL_LOSS = []
    TEST_ALL_IOU = []
    TEST_ALL_B_IOU = []
    LR = []
    EPOCH = []

    for epoch in range(epoch_num):
        print("_" * 15, epoch + 1, "_" * 15)
        print("learning_rate", Optimizer.state_dict()['param_groups'][0]['lr'])

        EPOCH.append(epoch + 1)
        LR.append(Optimizer.state_dict()['param_groups'][0]['lr'])
        # 训练过程
        train_mean_iou, train_mean_b_iou, train_mean_loss = train(model, train_data_loader, Optimizer)
        # 测试过程
        test_mean_iou, test_mean_b_iou, test_mean_loss = test(model, test_data_loader)
        # 记录数据
        TRAIN_ALL_IOU.append(train_mean_iou)
        TRAIN_ALL_B_IOU.append(train_mean_b_iou)
        TRAIN_ALL_LOSS.append(train_mean_loss)

        TEST_ALL_IOU.append(test_mean_iou)
        TEST_ALL_B_IOU.append(test_mean_b_iou)
        TEST_ALL_LOSS.append(test_mean_loss)

        #保存指标
        if test_mean_iou > 0.88:  #可改为0.8
            best_iou = test_mean_iou
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(model, 'Unet.pth')
            break_sign = 0
        if test_mean_b_iou > 0.6: #可改为0.4
            best_boundary_iou = test_mean_b_iou
            torch.save(model.state_dict(), 'best_biou.pth')
            break_sign = 0

        break_sign += 1
        print("best_iou:", best_iou, "best boundary iou:", best_boundary_iou)
        print("-" * 34)


        if early_stopping > 0 and break_sign > early_stopping and best_iou > signal_iou:
            print("达到要求，停止训练")
            break

        my_lr_scheduler.step()
        torch.cuda.empty_cache()


        if (epoch + 1) % 10 == 0:
            save_fig(plot_all,opt,TRAIN_ALL_IOU, TRAIN_ALL_LOSS, TEST_ALL_IOU, TEST_ALL_LOSS, LR, EPOCH, TRAIN_ALL_B_IOU,
                     TEST_ALL_B_IOU)
            show_predict(opt,save_num,test_data_loader, 0)
        torch.cuda.empty_cache()

    save_fig(plot_all,opt,TRAIN_ALL_IOU, TRAIN_ALL_LOSS, TEST_ALL_IOU, TEST_ALL_LOSS, LR, EPOCH, TRAIN_ALL_B_IOU, TEST_ALL_B_IOU)
    show_predict(opt,save_num,test_data_loader, 0)


if __name__ == '__main__':
    print("START")
    print("-" * 33)
    opt = parse_opt()
    main(opt)

