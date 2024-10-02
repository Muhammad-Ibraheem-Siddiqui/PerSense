import time
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

import timm

import os

import cv2

# assert "0.4.5" <= timm.__version__ <= "0.4.9"  # version check

from CounTR.util.misc import make_grid
from CounTR import models_mae_cross

device = torch.device('cuda')

"""
python demo.py
"""


class measure_time(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, typ, value, traceback):
        self.duration = (time.perf_counter_ns() - self.start) / 1e9


def load_image(test_image_path):
    # im_dir = '/l/users/muhammad.siddiqui/PerSense_Final/data/Images/Eggs'
    # im_id = '01.jpg'

    image = Image.open(test_image_path)
    image.load()
    W, H = image.size

    # Resize the image size so that the height is 384
    new_H = 384
    new_W = 16 * int((W / H * 384) / 16)
    # new_W = 3 * int((W / H * 384))
    # if new_W < 384:
    #     new_W = 384
    scale_factor_H = float(new_H) / H
    scale_factor_W = float(new_W) / W
    image = transforms.Resize((new_H, new_W))(image)



    Normalize = transforms.Compose([transforms.ToTensor()])
    image = Normalize(image)

    # If the width is less than 384, pad the image to make it 384 wide
    if new_W < 384:
        padding = (384 - new_W) // 2
        image = transforms.Pad((padding, 0, padding, 0))(image)
    
    # img_orig = image

    # Coordinates of the exemplar bound boxes
    # The left upper corner and the right lower corner

    # with open("./DSALVANet/test_data/bbox.txt", "r") as f:
    #             lines = f.readlines()
    #             bboxes = []
    #             for line in lines:
    #                 data = line.split()
    #                 bboxes.append(list(map(int,data[0:4])))

    with open("./DSALVANet/test_data/bbox.txt", "r") as f:
        lines = f.readlines()
        bboxes = []
        ori_boxes = []
        for line in lines:
            data = line.split()
            x1, y1, x2, y2 = map(int, data[0:4])
            bboxes.append([[y1, x1], [y2, x2]])
            ori_boxes.append(list(map(int,data[0:4])))
    # bboxes = [
    #     [[292, 172], [361, 226]],  #172 292 226 361
    #     [[517, 249], [572, 318]]   #249 517 318 572
    # ]
    # ori_boxes = bboxes
    boxes = list()
    rects = list()
    for bbox in bboxes:
        x1 = int(bbox[0][0] * scale_factor_W)
        y1 = int(bbox[0][1] * scale_factor_H)
        x2 = int(bbox[1][0] * scale_factor_W)
        y2 = int(bbox[1][1] * scale_factor_H)
        rects.append([y1, x1, y2, x2])
        # bbox = image[:, y1:y2, x1:x2]
        bbox = image[:, y1:y2 + 1, x1:x2 + 1]
        bbox = transforms.Resize((64, 64))(bbox)
        boxes.append(bbox.numpy())

    boxes = np.array(boxes)
    boxes = torch.Tensor(boxes)

    if boxes.shape[1] == 4:
        boxes = boxes[:, :3, :, :]


    return image, boxes, ori_boxes, rects, W, H, new_W, new_H


def run_one_image(samples, boxes, pos, model, W, H, test_idx, new_W, new_H):
    _, _, h, w = samples.shape

    s_cnt = 0
    for rect in pos:
        if rect[2] - rect[0] < 10 and rect[3] - rect[1] < 10:
            s_cnt += 1
    if s_cnt >= 1:
        r_densities = []
        r_images = []
        r_images.append(TF.crop(samples[0], 0, 0, int(h / 3), int(w / 3)))  # 1
        r_images.append(TF.crop(samples[0], 0, int(w / 3), int(h / 3), int(w / 3)))  # 3
        r_images.append(TF.crop(samples[0], 0, int(w * 2 / 3), int(h / 3), int(w / 3)))  # 7
        r_images.append(TF.crop(samples[0], int(h / 3), 0, int(h / 3), int(w / 3)))  # 2
        r_images.append(TF.crop(samples[0], int(h / 3), int(w / 3), int(h / 3), int(w / 3)))  # 4
        r_images.append(TF.crop(samples[0], int(h / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 8
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), 0, int(h / 3), int(w / 3)))  # 5
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w / 3), int(h / 3), int(w / 3)))  # 6
        r_images.append(TF.crop(samples[0], int(h * 2 / 3), int(w * 2 / 3), int(h / 3), int(w / 3)))  # 9

        pred_cnt = 0
        with measure_time() as et:
            for r_image in r_images:
                r_image = transforms.Resize((h, w))(r_image).unsqueeze(0)
                density_map = torch.zeros([h, w])
                density_map = density_map.to(device, non_blocking=True)
                start = 0
                prev = -1
                with torch.no_grad():
                    while start + 383 < w:
                        output, = model(r_image[:, :, :, start:start + 384], boxes, 3)
                        output = output.squeeze(0)
                        b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                        d1 = b1(output[:, 0:prev - start + 1])
                        b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                        d2 = b2(output[:, prev - start + 1:384])

                        b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                        density_map_l = b3(density_map[:, 0:start])
                        density_map_m = b1(density_map[:, start:prev + 1])
                        b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                        density_map_r = b4(density_map[:, prev + 1:w])

                        density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2
                        
                        # density_pred = density_map.permute(1, 2, 0)  # c x h x w -> h x w x c
                        density_pred = density_map.cpu().detach().numpy()
                        # Remove zero padding
                        # density_pred = remove_zero_padding(density_pred)
                        if new_W < 384:
                            crop_dim = (384 - new_W) // 2
                            density_pred = density_pred[0:384, crop_dim : crop_dim + new_W]
                        density_pred = cv2.resize(density_pred, (W, H))
                        density_pred = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
                        density_pred= np.stack((density_pred,) * 3, axis=-1)


                        # density_pred = cv2.cvtColor(density_pred, cv2.COLOR_RGB2BGR)
                        # density_map_path = "./outputs/" + "density_map"
                        # if not os.path.exists(density_map_path):
                        #         os.mkdir('./outputs/density_map')
                        # density_map_path = os.path.join(density_map_path, f'{test_idx}.png')
                        # cv2.imwrite(density_map_path,255*density_pred)


                        prev = start + 383
                        start = start + 128
                        if start + 383 >= w:
                            if start == w - 384 + 128:
                                break
                            else:
                                start = w - 384

                pred_cnt += torch.sum(density_map / 60).item()
                r_densities += [density_map]
    else:
        density_map = torch.zeros([h, w])
        density_map = density_map.to(device, non_blocking=True)
        start = 0
        prev = -1
        with measure_time() as et:
            with torch.no_grad():
                while start + 383 < w:
                    output, = model(samples[:, :, :, start:start + 384], boxes, 3)
                    output = output.squeeze(0)
                    b1 = nn.ZeroPad2d(padding=(start, w - prev - 1, 0, 0))
                    d1 = b1(output[:, 0:prev - start + 1])
                    b2 = nn.ZeroPad2d(padding=(prev + 1, w - start - 384, 0, 0))
                    d2 = b2(output[:, prev - start + 1:384])

                    b3 = nn.ZeroPad2d(padding=(0, w - start, 0, 0))
                    density_map_l = b3(density_map[:, 0:start])
                    density_map_m = b1(density_map[:, start:prev + 1])
                    b4 = nn.ZeroPad2d(padding=(prev + 1, 0, 0, 0))
                    density_map_r = b4(density_map[:, prev + 1:w])

                    density_map = density_map_l + density_map_r + density_map_m / 2 + d1 / 2 + d2

                    # density_map = density_map.permute(1, 2, 0)  # c x h x w -> h x w x c
                    density_pred = density_map.cpu().detach().numpy()
                    # Remove zero padding
                    # density_pred = remove_zero_padding(density_pred)
                    if new_W < 384:
                            crop_dim = (384 - new_W) // 2 #crop to modified size and then resize to original image size for overlaying
                            density_pred = density_pred[0:384, crop_dim : crop_dim + new_W]
                    density_pred = cv2.resize(density_pred, (W, H))
                    density_pred = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
                    density_pred= np.stack((density_pred,) * 3, axis=-1)


                    # density_pred = cv2.cvtColor(density_pred, cv2.COLOR_RGB2BGR)
                    # density_map_path = "./outputs/" + "density_map"
                    # if not os.path.exists(density_map_path):
                    #         os.mkdir('./outputs/density_map')
                    # density_map_path = os.path.join(density_map_path, f'{test_idx}.png')
                    # cv2.imwrite(density_map_path,255*density_pred)

                    prev = start + 383
                    start = start + 128
                    if start + 383 >= w:
                        if start == w - 384 + 128:
                            break
                        else:
                            start = w - 384

            pred_cnt = torch.sum(density_map / 60).item()

    e_cnt = 0
    for rect in pos:
        e_cnt += torch.sum(density_map[rect[0]:rect[2] + 1, rect[1]:rect[3] + 1] / 60).item()
    e_cnt = e_cnt / 3
    if e_cnt > 1.8:
        pred_cnt /= e_cnt

    # Visualize the prediction
    # fig = samples[0]
    # box_map = torch.zeros([fig.shape[1], fig.shape[2]])
    # box_map = box_map.to(device, non_blocking=True)
    # for rect in pos:
    #     for i in range(rect[2] - rect[0]):
    #         box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[1], fig.shape[2] - 1)] = 10
    #         box_map[min(rect[0] + i, fig.shape[1] - 1), min(rect[3], fig.shape[2] - 1)] = 10
    #     for i in range(rect[3] - rect[1]):
    #         box_map[min(rect[0], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
    #         box_map[min(rect[2], fig.shape[1] - 1), min(rect[1] + i, fig.shape[2] - 1)] = 10
    # box_map = box_map.unsqueeze(0).repeat(3, 1, 1)
    # pred = density_map.unsqueeze(0).repeat(3, 1, 1) if s_cnt < 1 \
    #     else make_grid(r_densities, h, w).unsqueeze(0).repeat(3, 1, 1)
    # fig = fig + box_map + pred / 2
    # fig = torch.clamp(fig, 0, 1)
    # torchvision.utils.save_image(fig, f'./CounTR/Image/Visualisation.png')
    # GT map needs coordinates for all GT dots, which is hard to input and is not a must for the demo. You can provide it yourself.
    return pred_cnt, et, density_pred
# # Prepare model
# model = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
# model.to(device)
# model_without_ddp = model

# checkpoint = torch.load('./CounTR/output_allnew_dir/FSC147.pth', map_location='cpu')
# model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
# print("Resume checkpoint %s" % './output_allnew_dir/FSC147.pth')

# model.eval()

# # Test on the new image
# samples, boxes, pos, W, H = load_image(src_img)
# samples = samples.unsqueeze(0).to(device, non_blocking=True)
# boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

# result, elapsed_time, density_pred = run_one_image(samples, boxes, pos, model, W, H)
# print(result, elapsed_time.duration)
