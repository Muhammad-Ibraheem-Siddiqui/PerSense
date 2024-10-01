r""" Matcher testing code for one-shot segmentation """
import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np

import sys
sys.path.append('./')

from matcher.common.logger import Logger, AverageMeter
from matcher.common.vis import Visualizer
from matcher.common.evaluation import Evaluator
from matcher.common import utils
from matcher.data.dataset import FSSDataset
from matcher.Matcher import build_matcher_oss

import cv2

import random
random.seed(0)

import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

def test(matcher, obj_name, output_path, images_path, masks_path, args):
    r""" Test Matcher """

    print("\n------------> Segment " + obj_name)

    # Path preparation
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    # Load images and masks
    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    ref_image = torch.tensor(ref_image)
    ref_image = ref_image.permute(2,0,1)
    size = (518,518)
    transform = transforms.Resize(size)
    ref_image = transform(ref_image)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
    ref_mask = torch.tensor(ref_mask)
    ref_mask = ref_mask.permute(2,0,1)
    size = (518,518)
    transform = transforms.Resize(size)
    ref_mask = transform(ref_mask)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

    loop_over = len(os.listdir(test_images_path))
    for test_idx in tqdm(range(loop_over//2)):

        # ref_image = cv2.imread("/l/users/muhammad.siddiqui/Matcher/datasets/Lemons/00.jpg")
        # ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
        # ref_image = torch.tensor(ref_image)
        # ref_image = ref_image.permute(2,0,1)
        # size = (518,518)
        # transform = transforms.Resize(size)
        # ref_image = transform(ref_image)
        
        # ref_mask = cv2.imread("/l/users/muhammad.siddiqui/Matcher/datasets/Lemons/00.png")
        # ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)
        # ref_mask = torch.tensor(ref_mask)
        # ref_mask = ref_mask.permute(2,0,1)
        # size = (518,518)
        # transform = transforms.Resize(size)
        # ref_mask = transform(ref_mask)

        # gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
        # gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()

        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        original_image = test_image

        # test_image = cv2.imread("/l/users/muhammad.siddiqui/Matcher/datasets/Lemons/01.jpg")
        # test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        test_image = torch.tensor(test_image)
        test_image = test_image.permute(2,0,1)
        size = (518,518)
        transform = transforms.Resize(size)
        test_image = transform(test_image)
        

        # test_mask = cv2.imread("/l/users/muhammad.siddiqui/Matcher/datasets/Lemons/01.jpg")
        # test_mask = cv2.cvtColor(test_mask, cv2.COLOR_BGR2RGB)
        # test_mask = torch.tensor(test_mask)
        # test_mask = test_mask.permute(2,0,1)
        # size = (518,518)
        # transform = transforms.Resize(size)
        # test_mask = transform(test_mask)



        # 1. Matcher prepare references and target
        matcher.set_reference(ref_image, ref_mask)
        matcher.set_target(test_image)

        # 2. Predict mask of target
        pred_mask = matcher.predict().squeeze(0)
        target_height, target_width, target_channels = original_image.shape[0], original_image.shape[1], original_image.shape[2]

        # Resize the mask to original image size
        pred_mask = pred_mask.unsqueeze(0)
        pred_mask = F.interpolate(pred_mask.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False)
        pred_mask = pred_mask.squeeze(0).permute(1, 2, 0)  # permute to (H, W, C)
        mask_show = pred_mask.cpu().detach().numpy()
        # plt.figure(figsize=(10, 10))
        # plt.imshow(mask_show)


        # Normalize mask to be in range [0, 1]
        mask_show = (mask_show - mask_show.min()) / (mask_show.max() - mask_show.min())

        # Convert mask to 3 channels if it's single channel
        if mask_show.shape[2] == 1:
            mask_show = cv2.cvtColor(mask_show, cv2.COLOR_GRAY2RGB)

        # Overlay mask on original image
        alpha = 0.8  # Transparency factor

        # Ensure mask_show is in the same dtype and range as original_image
        mask_show = (mask_show * 255).astype(np.uint8)

        mask_indices = mask_show[:, :, 0] > 0  # Create a boolean mask where mask_show is greater than zero
        overlay_image = original_image.copy()
        overlay_image[mask_indices] = cv2.addWeighted(original_image, 1 - alpha, mask_show, alpha, 0)[mask_indices]

        # # Create the overlay
        # overlay_image = cv2.addWeighted(original_image, 1 - alpha, mask_show, alpha, 0)

        # Convert back to BGR for saving or displaying using OpenCV
        overlay_image_bgr = cv2.cvtColor(overlay_image, cv2.COLOR_BGR2RGB)

        # Save or display the result
        # vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        # cv2.imwrite(vis_mask_output_path, overlay_image_bgr)

        # cv2.imwrite("overlay_image.jpg", overlay_image_bgr)
        # cv2.imshow("Overlay Image", overlay_image_bgr)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        

        mask_output_path = os.path.join(output_path, f'{test_idx}.png')
        cv2.imwrite(mask_output_path, pred_mask.cpu().numpy().astype(np.uint8) * 255)

        matcher.clear()


if __name__ == '__main__':

    # Arguments parsing
    parser = argparse.ArgumentParser(description='Matcher Pytorch Implementation for One-shot Segmentation')

    # Dataset parameters
    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--datapath', type=str, default='datasets')
    parser.add_argument('--benchmark', type=str, default='coco',
                        choices=['fss', 'coco', 'pascal', 'lvis', 'paco_part', 'pascal_part'])
    parser.add_argument('--bsz', type=int, default=1)
    parser.add_argument('--nworker', type=int, default=0)
    parser.add_argument('--fold', type=int, default=0)
    parser.add_argument('--nshot', type=int, default=1)
    parser.add_argument('--img-size', type=int, default=518)
    parser.add_argument('--use_original_imgsize', action='store_true')
    parser.add_argument('--log-root', type=str, default='output/debug')
    parser.add_argument('--visualize', type=int, default=0)
    parser.add_argument('--ref_idx', type=str, default='00')

    # DINOv2 and SAM parameters
    parser.add_argument('--dinov2-size', type=str, default="vit_large")
    parser.add_argument('--sam-size', type=str, default="vit_h")
    parser.add_argument('--dinov2-weights', type=str, default="models/dinov2_vitl14_pretrain.pth")
    parser.add_argument('--sam-weights', type=str, default="models/sam_vit_h_4b8939.pth")
    parser.add_argument('--use_semantic_sam', action='store_true', help='use semantic-sam')
    parser.add_argument('--semantic-sam-weights', type=str, default="models/swint_only_sam_many2many.pth")
    parser.add_argument('--points_per_side', type=int, default=64)
    parser.add_argument('--pred_iou_thresh', type=float, default=0.88)
    parser.add_argument('--sel_stability_score_thresh', type=float, default=0.0)
    parser.add_argument('--stability_score_thresh', type=float, default=0.95)
    parser.add_argument('--iou_filter', type=float, default=0.0)
    parser.add_argument('--box_nms_thresh', type=float, default=1.0)
    parser.add_argument('--output_layer', type=int, default=3)
    parser.add_argument('--dense_multimask_output', type=int, default=0)
    parser.add_argument('--use_dense_mask', type=int, default=0)
    parser.add_argument('--multimask_output', type=int, default=0)

    # Matcher parameters
    parser.add_argument('--num_centers', type=int, default=8, help='K centers for kmeans')
    parser.add_argument('--use_box', action='store_true', help='use box as an extra prompt for sam')
    parser.add_argument('--use_points_or_centers', action='store_true', help='points:T, center: F')
    parser.add_argument('--sample-range', type=str, default="(4,6)", help='sample points number range')
    parser.add_argument('--max_sample_iterations', type=int, default=30)
    parser.add_argument('--alpha', type=float, default=1.)
    parser.add_argument('--beta', type=float, default=0.)
    parser.add_argument('--exp', type=float, default=0.)
    parser.add_argument('--emd_filter', type=float, default=0.0, help='use emd_filter')
    parser.add_argument('--purity_filter', type=float, default=0.0, help='use purity_filter')
    parser.add_argument('--coverage_filter', type=float, default=0.0, help='use coverage_filter')
    parser.add_argument('--use_score_filter', action='store_true')
    parser.add_argument('--deep_score_norm_filter', type=float, default=0.1)
    parser.add_argument('--deep_score_filter', type=float, default=0.33)
    parser.add_argument('--topk_scores_threshold', type=float, default=0.7)
    parser.add_argument('--num_merging_mask', type=int, default=10, help='topk masks for merging')
    parser.add_argument('--outdir', type=str, default='Matcher')

    args = parser.parse_args()
    args.sample_range = eval(args.sample_range)

    if not os.path.exists(args.log_root):
        os.makedirs(args.log_root)

    Logger.initialize(args, root=args.log_root)

    # Device setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.device = device
    Logger.info('# available GPUs: %d' % torch.cuda.device_count())

    # Model initialization
    if not args.use_semantic_sam:
        matcher = build_matcher_oss(args)
    else:
        from matcher.Matcher_SemanticSAM import build_matcher_oss as build_matcher_semantic_sam_oss
        matcher = build_matcher_semantic_sam_oss(args)

    # Helper classes (for testing) initialization
    Evaluator.initialize()
    Visualizer.initialize(args.visualize)

    # Dataset initialization
    # FSSDataset.initialize(img_size=args.img_size, datapath=args.datapath, use_original_imgsize=args.use_original_imgsize)
    # dataloader_test = FSSDataset.build_dataloader(args.benchmark, args.bsz, args.nworker, args.fold, 'test', args.nshot)

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')

    images_path = args.data + '/Images/'
    output_path = './outputs/' + args.outdir
    masks_path = args.data + '/Images/'

    # Test Matcher
    with torch.no_grad():
        for obj_name in os.listdir(images_path):
            test(matcher, obj_name, output_path, images_path, masks_path, args=args)