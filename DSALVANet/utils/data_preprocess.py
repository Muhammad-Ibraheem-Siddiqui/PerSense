import cv2
import torchvision.transforms as transforms
import torch

def preprocess(ori_img, ori_boxes, device):
    h_rsz,w_rsz = 512,512
    ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
    h_orig,w_orig = ori_img.shape[:2]
    rsz_image = cv2.resize(ori_img, (h_rsz, w_rsz))
    h_scale, w_scale = h_rsz / h_orig, w_rsz / w_orig
    if len(ori_boxes) >3:
         ori_boxes = ori_boxes[:3]
    for i in range(3-len(ori_boxes)):
            ori_boxes.append(ori_boxes[i])
    rsz_boxes = []
    for box in ori_boxes:
        y_tl, x_tl, y_br, x_br = box
        y_tl = int(y_tl * h_scale)
        y_br = int(y_br * h_scale)
        x_tl = int(x_tl * w_scale)
        x_br = int(x_br * w_scale)
        rsz_boxes.append([y_tl, x_tl, y_br, x_br])
    rsz_image = transforms.ToTensor()(rsz_image)
    rsz_boxes = torch.tensor(rsz_boxes, dtype=torch.float64).unsqueeze(0).to(device)
    normalize_fn = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    rsz_image = normalize_fn(rsz_image).unsqueeze(0).to(device)
    return rsz_image, rsz_boxes