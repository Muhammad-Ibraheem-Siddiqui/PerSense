import os
import cv2
import numpy as np
from pathlib import Path
import argparse
from matplotlib import pyplot as plt



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_path', type=str, default='PerSense') #'baseline' for evaluating baseline results and 'PerSense' for evaluating persense
    parser.add_argument('--gt_path', type=str, default='./data/Images')

    parser.add_argument('--ref_idx', type=str, default='00')
    
    args = parser.parse_args()
    return args

all_class_names = []
miou_values = []

def main():

    args = get_arguments()
    print("Args:", args, "\n"), 

    class_names = sorted(os.listdir(args.gt_path))
    class_names = [class_name for class_name in class_names if ".DS" not in class_name]
    class_names.sort()

    mIoU, mAcc = 0, 0
    count = 0
    for class_name in class_names:
        count += 1
        gt_path_class = os.path.join(args.gt_path, class_name)
        pred_path_class = os.path.join("./outputs/" + args.pred_path, class_name)

        gt_images = [str(img_path) for img_path in sorted(Path(gt_path_class).rglob("*.png"))]
        pred_images = [str(img_path) for img_path in sorted(Path(pred_path_class).rglob("*.png"))]

        intersection_meter = AverageMeter()
        union_meter = AverageMeter()
        target_meter = AverageMeter()

        for i, (gt_img, pred_img) in enumerate(zip(gt_images, pred_images)): 
            if args.ref_idx in gt_img:
                continue

            gt_img = cv2.imread(gt_img)
            gt_img = cv2.cvtColor(gt_img, cv2.COLOR_BGR2GRAY) > 0
            gt_img = np.uint8(gt_img)

            pred_img = cv2.imread(pred_img)
            pred_img = cv2.cvtColor(pred_img, cv2.COLOR_BGR2GRAY) > 0
            pred_img = np.uint8(pred_img)

            intersection, union, target = intersectionAndUnion(pred_img, gt_img)
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

        iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)

        print(class_name + ',', "IoU: %.2f," %(100 * iou_class))

        mIoU += iou_class

        all_class_names.append(class_name)
        miou_values.append(iou_class)

    # # Plotting
    # plt.figure(figsize=(10, 6))
    # plt.bar(class_names, miou_values, color='skyblue')
    # plt.xlabel('Class Names')
    # plt.ylabel('mIoU')
    # plt.title('mIoU per Class')
    # plt.xticks(rotation=45, ha='right')
    # plt.tight_layout()

    # dist_path = './outputs/' + 'class_wise_miou/'
    # os.makedirs(dist_path, exist_ok=True) 
    # dist_output_path = os.path.join(dist_path,'' f'class_wise_miou.png')
    # with open(dist_output_path, 'wb') as outfile:
    #         plt.savefig(outfile, format='png')
    # plt.close()



    print("\nmIoU: %.2f" %(100 * mIoU / count))

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


def intersectionAndUnion(output, target):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    
    area_intersection = np.logical_and(output, target).sum()
    area_union = np.logical_or(output, target).sum()
    area_target = target.sum()
    
    return area_intersection, area_union, area_target


if __name__ == '__main__':
    main()
