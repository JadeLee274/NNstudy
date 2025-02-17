import argparse

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image
import os

from models import SRCNN
from utils import convert_rgb_to_ycbcr, convert_ycbcr_to_rgb, calc_psnr

if __name__ == '__main__':
    root = '/data/home/tmdals274/NNstudy/SRCNN/SRCNN-practice/outputs/x3'
    weight_file = os.path.join(root, 'epoch_400.pth')
    image_file = '/data/home/tmdals274/NNstudy/SRCNN/SRCNN-practice/data/butterfly_GT.bmp'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type = str, defalut = weight_file)
    parser.add_argument('--image-file', type = str, default = image_file)
    parser.add_argument('--scale', type = int, default = 3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()

    for n, p in torch.load(args.weight_file, map_location = lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)
        
    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale

    image = image.resize((image_width, image_height), resample = pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample = pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample = pil_image.BICUBIC)

    image.save(args.image_file.replace('.', f'_bicubic_x{args.scale}.'))
    
    image = np.array(image).astype(np.float32)
    ycbcr = convert_ycbcr_to_rgb(image)

    y = ycbcr[..., 0]
    y /= 255.
    y = torch.from_numpy(y).to(device)
    y = y.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(y).clamp(0.0, 1.0)

    psnr = preds.mul(255.0).cpu().numpy().squeeze(0).squeeze(0)

    output = np.array([preds, ycbcr[..., 1], ycbcr[..., 2]]).transpose([1, 2, 0])
    output = np.clip(convert_ycbcr_to_rgb(output), 0.0, 255.0).astype(np.unit8)
    outpout = pil_image.fromarray(output)
    output.save(args.image_file.replace('.', f'_srcnn_x{args.scale}.'))