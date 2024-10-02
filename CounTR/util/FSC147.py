from argparse import Namespace
import json
from pathlib import Path

import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image

import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]


class ResizeSomeImage(object):
    def __init__(self, args):
        self.im_dir = args.im_dir
        anno_file = args.anno_file
        data_split_file = args.data_split_file

        with open(anno_file) as f:
            self.annotations = json.load(f)

        with open(data_split_file) as f:
            data_split = json.load(f)

        self.train_set = data_split['train']

        self.class_dict = {}
        if args.do_aug:
            with open(args.class_file) as f:
                for line in f:
                    key = line.split()[0]
                    val = line.split()[1:]
                    self.class_dict[key] = val


class ResizePreTrainImage(ResizeSomeImage):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is preserved
    Density and boxes correctness not preserved(crop and horizontal flip)
    """

    def __init__(self, args, MAX_HW=384):
        super().__init__(args)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, lines_boxes, density = sample['image'], sample['lines_boxes'], sample['gt_density']

        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0:
            resized_density = resized_density * (orig_count / new_count)

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1, x1, y2, x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image': resized_image, 'boxes': boxes, 'gt_density': resized_density}
        return sample


class ResizeTrainImage(ResizeSomeImage):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """

    def __init__(self, args, MAX_HW=384, do_aug=True):
        super().__init__(args)
        self.max_hw = MAX_HW
        self.do_aug = do_aug

    def __call__(self, sample):
        image, lines_boxes, dots, im_id, m_flag = sample['image'], sample['lines_boxes'], \
            sample['dots'], sample['id'], sample['m_flag']

        W, H = image.size

        new_H = 16 * int(H / 16)
        new_W = 16 * int(W / 16)
        scale_factor_h = float(new_H) / H
        scale_factor_w = float(new_W) / W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_image = TTensor(resized_image)
        resized_density = np.zeros((new_H, new_W), dtype='float32')

        # Augmentation probability
        aug_flag = self.do_aug
        mosaic_flag = random.random() < 0.25

        if aug_flag:
            # Gaussian noise
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

            # Color jitter and Gaussian blur
            re_image = Augmentation(re_image)

            # Random affine
            re1_image = re_image.transpose(0, 1).transpose(1, 2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W - 1, int(dots[i][0] * scale_factor_w)), y=min(new_H - 1, int(dots[i][1] * scale_factor_h))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15, 15),
                    scale=(0.8, 1.2),
                    shear=(-10, 10),
                    translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]), dtype='float32')
            for i in range(len(kps.keypoints)):
                if (int(kps_aug.keypoints[i].y) <= new_H - 1 and int(kps_aug.keypoints[i].x) <= new_W - 1) and not \
                        kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)] = 1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

            # Random horizontal flip
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)    

            # Random self mosaic
            if mosaic_flag:
                image_array = []
                map_array = []
                blending_l = random.randint(10, 20)
                resize_l = 192 + 2 * blending_l
                if dots.shape[0] >= 70:
                    for i in range(4):
                        length = random.randint(150, 384)
                        start_W = random.randint(0, new_W - length)
                        start_H = random.randint(0, new_H - length)
                        reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                        reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                        reresized_density1 = np.zeros((resize_l, resize_l), dtype='float32')
                        for i in range(dots.shape[0]):
                            if start_H <= min(new_H - 1, int(dots[i][1] * scale_factor_h)) < start_H + length and start_W <= min(new_W - 1, int(dots[i][0] * scale_factor_w)) < start_W + length:
                                reresized_density1[min(resize_l-1,int((min(new_H-1,int(dots[i][1] * scale_factor_h))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_W-1,int(dots[i][0] * scale_factor_w))-start_W)*resize_l/length))]=1
                        reresized_density1 = torch.from_numpy(reresized_density1)
                        image_array.append(reresized_image1)
                        map_array.append(reresized_density1)
                else:
                    m_flag = 1
                    prob = random.random()
                    if prob > 0.25:
                        gt_pos = random.randint(0, 3)
                    else:
                        gt_pos = random.randint(0, 4)  # 5% 0 objects
                    for i in range(4):
                        if i == gt_pos:
                            Tim_id = im_id
                            r_image = resized_image
                            Tdots = dots
                            new_TH = new_H
                            new_TW = new_W
                            Tscale_factor_w = scale_factor_w
                            Tscale_factor_h = scale_factor_h
                        else:
                            Tim_id = self.train_set[random.randint(0, len(self.train_set) - 1)]
                            Tdots = np.array(self.annotations[Tim_id]['points'])
                            Timage = Image.open('{}/{}'.format(self.im_dir, Tim_id))
                            Timage.load()
                            new_TW = 16 * int(Timage.size[0] / 16)
                            new_TH = 16 * int(Timage.size[1] / 16)
                            Tscale_factor_w = float(new_TW) / Timage.size[0]
                            Tscale_factor_h = float(new_TH) / Timage.size[1]
                            r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                        length = random.randint(250, 384)
                        start_W = random.randint(0, new_TW - length)
                        start_H = random.randint(0, new_TH - length)
                        r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                        r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                        r_density1 = np.zeros((resize_l, resize_l), dtype='float32')
                        if self.class_dict[im_id] == self.class_dict[Tim_id]:
                            for i in range(Tdots.shape[0]):
                                if start_H <= min(new_TH - 1, int(Tdots[i][1] * Tscale_factor_h)) < start_H + length and start_W <= min(new_TW - 1, int(Tdots[i][0] * Tscale_factor_w)) < start_W + length:
                                    r_density1[min(resize_l-1,int((min(new_TH-1, int(Tdots[i][1] * Tscale_factor_h))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_TW-1,int(Tdots[i][0] * Tscale_factor_w))-start_W)*resize_l/length))]=1
                        r_density1 = torch.from_numpy(r_density1)
                        image_array.append(r_image1)
                        map_array.append(r_density1)

                reresized_image5 = torch.cat((image_array[0][:, blending_l:resize_l-blending_l], image_array[1][:, blending_l: resize_l-blending_l]), 1)
                reresized_density5 = torch.cat((map_array[0][blending_l:resize_l-blending_l], map_array[1][blending_l: resize_l-blending_l]), 0)
                for i in range(blending_l):
                        reresized_image5[:, 192+i] = image_array[0][:, resize_l-1-blending_l+i] * (blending_l-i)/(2 * blending_l) + reresized_image5[:, 192+i] * (i+blending_l)/(2*blending_l)
                        reresized_image5[:, 191-i] = image_array[1][:, blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:, 191-i] * (i+blending_l)/(2*blending_l)
                reresized_image5 = torch.clamp(reresized_image5, 0, 1)

                reresized_image6 = torch.cat((image_array[2][:, blending_l:resize_l-blending_l], image_array[3][:, blending_l: resize_l-blending_l]), 1)
                reresized_density6 = torch.cat((map_array[2][blending_l:resize_l-blending_l], map_array[3][blending_l:resize_l-blending_l]), 0)
                for i in range(blending_l):
                        reresized_image6[:, 192+i] = image_array[2][:, resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:, 192+i] * (i+blending_l)/(2*blending_l)
                        reresized_image6[:, 191-i] = image_array[3][:, blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:, 191-i] * (i+blending_l)/(2*blending_l)
                reresized_image6 = torch.clamp(reresized_image6, 0, 1)

                reresized_image = torch.cat((reresized_image5[:, :, blending_l:resize_l-blending_l], reresized_image6[:, :, blending_l:resize_l-blending_l]), 2)
                reresized_density = torch.cat((reresized_density5[:, blending_l:resize_l-blending_l], reresized_density6[:, blending_l:resize_l-blending_l]), 1)
                for i in range(blending_l):
                        reresized_image[:, :, 192+i] = reresized_image5[:, :, resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:, :, 192+i] * (i+blending_l)/(2*blending_l)
                        reresized_image[:, :, 191-i] = reresized_image6[:, :, blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:, :, 191-i] * (i+blending_l)/(2*blending_l)
                reresized_image = torch.clamp(reresized_image, 0, 1)

            else:
                # Random 384*384 crop in a new_W*384 image and 384*new_W density map
                start = random.randint(0, new_W - 1 - 383)
                reresized_image = TF.crop(re_image, 0, start, 384, 384)
                reresized_density = resized_density[:, start:start + 384]

        else:
            # Random 384*384 crop in a new_W*384 image and 384*new_W density map
            for i in range(dots.shape[0]):
                resized_density[min(new_H - 1, int(dots[i][1] * scale_factor_h))] \
                                [min(new_W - 1, int(dots[i][0] * scale_factor_w))] = 1
            resized_density = torch.from_numpy(resized_density)
            start = random.randint(0, new_W - self.max_hw)
            reresized_image = TF.crop(resized_image, 0, start, self.max_hw, self.max_hw)
            reresized_density = resized_density[0:self.max_hw, start:start + self.max_hw]

        # Gaussian distribution density map
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)

        # Density map scale up
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)

        # Crop bboxes and resize as 64x64
        boxes = list()
        rects = list()
        cnt = 0
        for box in lines_boxes:
            cnt += 1
            if cnt > 3:
                break
            box2 = [int(k) for k in box]
            y1 = int(box2[0] * scale_factor_h)
            x1 = int(box2[1] * scale_factor_w)
            y2 = int(box2[2] * scale_factor_h)
            x2 = int(box2[3] * scale_factor_w)
            if not aug_flag:
                rects.append(torch.tensor([y1, max(0, x1-start), y2, min(self.max_hw, x2-start)]))
            bbox = resized_image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox)
        boxes = torch.stack(boxes)
        if aug_flag:
            pos = torch.tensor([])
        else:
            pos = torch.stack(rects)

        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]
        sample = {'image': reresized_image, 'boxes': boxes, 'pos': pos, 'gt_density': reresized_density, 'm_flag': m_flag}

        return sample


class ResizeValImage(ResizeSomeImage):
    def __init__(self, args, MAX_HW=384):
        super().__init__(args)
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image, dots, m_flag, lines_boxes = sample['image'], sample['dots'], sample['m_flag'], sample['lines_boxes']

        W, H = image.size

        new_H = new_W = self.max_hw
        scale_factor_h = float(new_H) / H
        scale_factor_w = float(new_W) / W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_image = TTensor(resized_image)

        # Resize density map
        resized_density = np.zeros((new_H, new_W), dtype='float32')
        for i in range(dots.shape[0]):
            resized_density[min(new_H - 1, int(dots[i][1] * scale_factor_h))] \
                           [min(new_W - 1, int(dots[i][0] * scale_factor_w))] = 1
        resized_density = ndimage.gaussian_filter(resized_density, sigma=4, radius=7, order=0)
        resized_density = torch.from_numpy(resized_density) * 60

        # Crop bboxes and resize as 64x64
        boxes = list()
        rects = list()
        cnt = 0
        for box in lines_boxes:
            cnt += 1
            if cnt > 3:
                break
            box2 = [int(k) for k in box]
            y1 = int(box2[0] * scale_factor_h)
            x1 = int(box2[1] * scale_factor_w)
            y2 = int(box2[2] * scale_factor_h)
            x2 = int(box2[3] * scale_factor_w)
            rects.append(torch.tensor([y1, x1, y2, x2]))
            bbox = resized_image[:, y1:y2 + 1, x1:x2 + 1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox)
        boxes = torch.stack(boxes)
        pos = torch.stack(rects)

        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]
        sample = {'image': resized_image, 'boxes': boxes, 'pos': pos, 'gt_density': resized_density, 'm_flag': m_flag}
        return sample


PreTrainNormalize = transforms.Compose([
    transforms.RandomResizedCrop(MAX_HW, scale=(0.2, 1.0), interpolation=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
])

TTensor = transforms.Compose([
    transforms.ToTensor(),
])

Augmentation = transforms.Compose([
    transforms.ColorJitter(brightness=0.25, contrast=0.15, saturation=0.15, hue=0.15),
    transforms.GaussianBlur(kernel_size=(7, 9))
])

Normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
])


def transform_train(args: Namespace, do_aug=True):
    return transforms.Compose([ResizeTrainImage(args, MAX_HW, do_aug)])

def transform_val(args: Namespace):
    return transforms.Compose([ResizeValImage(args, MAX_HW)])

def transform_pre_train(args: Namespace):
    return transforms.Compose([ResizePreTrainImage(args, MAX_HW)])
