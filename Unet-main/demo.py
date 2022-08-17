'''
测试文件，可以通过该文件观察模型预测的边界和图形的效果
注意：该文件的权重文件、setting.txt需要修改至指定地址
'''
import torch
import numpy as np
from Models import U_Net,NestedUNet

from Dataset import HorseDataset
from torchvision import transforms
from augmentation import Transform_Compose, Train_Transform, Totensor, Test_Transform
from computeloss import  generate_boundary
from PIL import Image


idx = np.arange(327)
testing_idx = idx[278:327]


train_transforms = Transform_Compose([Train_Transform(image_size=80), Totensor()])
test_transforms = Transform_Compose([Test_Transform(image_size=80), Totensor()])

root = 'D:/Dataset/horse/weizmann_horse_db'
with open('Unetparam/setting.txt', 'r') as f:
    lines = f.readlines()
    f.close()

for line in lines:
    if "root" in line:
        line = line.rstrip("\n")
        line_split= line.split(' ')
        root = line_split[2]


print("-" * 30)
print("loading...")
# 载入测试数据
Test_data = HorseDataset(root, testing_idx, test_transforms)

test_data_loader = torch.utils.data.DataLoader(
    Test_data,
    batch_size=8,
    num_workers=0,
    shuffle=True,
    pin_memory=True,
    drop_last=True
)

if __name__ == '__main__':
    print("-" * 34)
    print("demo--strating")
    best_model = U_Net()
    state_dict = torch.load('Unetparam/best_model.pth')
    best_model.load_state_dict(state_dict, strict=False)

    toPIL = transforms.ToPILImage()

    for test_data, test_mask in test_data_loader:

        B, H, W = test_mask.shape
        pic1 = toPIL(test_data[0])
        pic2 = toPIL(test_mask[0])
        print("图像依次为原图，mask,mask边界，预测图像和预测边界")
        pic1.show()
        pic2.show()
        outputs = best_model(test_data)
        mask_boundary = 255 * generate_boundary(test_mask, dilation_ratio=0.02, sign=1)


        out_boundary = 255 * generate_boundary(outputs[-1].squeeze(dim=1), dilation_ratio=0.02, sign=1)
        out = outputs[-1].squeeze(dim=1)

        Save_out = torch.sigmoid(out).data.cpu().numpy()
        Save_out[Save_out > 0.5] = 255
        Save_out[Save_out <= 0.5] = 0

        test_mask_ = torch.sigmoid(test_mask).data.cpu().numpy()
        test_mask_[test_mask_ > 0.5] = 255
        test_mask_[test_mask_ <= 0.5] = 0

        A = Image.fromarray(mask_boundary[0].astype('uint8'))
        B = Image.fromarray(out_boundary[0].astype('uint8'))
        A.show()

        X = Save_out[0].astype('uint8')
        Z = Image.fromarray(X)
        Z.show()
        B.show()
        print("end")
        print("-" * 34)
        break
