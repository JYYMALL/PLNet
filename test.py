import imageio
import torch
import torch.nn.functional as F
import numpy as np
import os, argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from net.plnet import PLNet
from utils.tdataloader import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=512, help='testing size')
parser.add_argument('--pth_path', type=str, default='./checkpoints/PLNet/PLNet-BEST.pth')

save_path = './results/PLNet/{}'.format("ttpla")
opt = parser.parse_args()
model = PLNet().cuda()
model.load_state_dict(torch.load(opt.pth_path))
model.eval()

os.makedirs(save_path, exist_ok=True)
os.makedirs(os.path.join(save_path, 'ht'), exist_ok=True)
image_root = '{}/'.format("data/test/images")
gt_root = '{}/'.format("data/test/mask")
test_loader = test_dataset(image_root, gt_root, opt.testsize)

for i in range(test_loader.size):
    image, gt, name = test_loader.load_data()
    gt = np.asarray(gt, np.float32)
    gt /= (gt.max() + 1e-8)
    gt[gt >= 0.5] = 1
    gt[gt < 0.5] = 0
    image = image.cuda()
    torch.cuda.synchronize()
    o4, o3, o2, res, ht = model(image)

    res = F.interpolate(res, size=gt.shape, mode='bilinear', align_corners=False)
    res = res.sigmoid().data.cpu().numpy().squeeze()
    res = (res - res.min()) / (res.max() - res.min() + 1e-8)
    att = ht.sigmoid().data.cpu().numpy().squeeze()
    imageio.imwrite(os.path.join(save_path, 'ht', name), (att * 255).astype(np.uint8))
    imageio.imwrite(os.path.join(save_path, name), (res * 255).astype(np.uint8))
