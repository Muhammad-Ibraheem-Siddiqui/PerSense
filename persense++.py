import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import warnings
warnings.filterwarnings('ignore')
from show import *
from per_segment_anything import sam_model_registry, SamPredictor
from DSALVANet.utils.PerSense_hybrid import hybrid_IDM
from DSALVANet.utils.data_preprocess import preprocess
from DSALVANet.utils.model_helper import build_model
from DSALVANet.utils.PerSense_countr import hybrid_IDM_countr
from ViPLLaVA.llava.model.builder import load_pretrained_model
from ViPLLaVA.llava.mm_utils import get_model_name_from_path
from ViPLLaVA.llava.eval.run_llava import eval_model
from PIL import Image , ImageChops
from torchvision import transforms
from torchvision.ops import roi_align
from GroundingDINO.groundingdino.util.inference import Model
from visualize_cv import cv2_visualization
from typing import List
from CounTR import models_mae_cross
from CounTR.demo import load_image, run_one_image
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torchvision.models as models
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
import gc



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='PerSense++')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--visualize', type=bool, default= True) # Change to True for visualization
    parser.add_argument('--fsoc', type=str, default='DSALVANet') #use 'DSALVANet' for DMG1 and 'countr' for DMG2 and 

    
    args = parser.parse_args()
    return args


def main():

    args = get_arguments()
    print("Args:", args)
    

    # Load ViPLLaVA model
    print("======> Load LLM" )
    model_path = "mucai/vip-llava-7b"
    model_name = get_model_name_from_path(model_path)
    model_base = None
    # model = load_pretrained_model(model_path).cuda()
    llava_tokenizer, model_llava, llava_image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )
    print("======> Done" )

    print("======> Load SAM" )
    if args.sam_type == 'vit_h':
        sam_type, sam_ckpt = 'vit_h', 'data/sam_vit_h_4b8939.pth'
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).cuda()
    elif args.sam_type == 'vit_t':
        sam_type, sam_ckpt = 'vit_t', 'weights/mobile_sam.pt'
        device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[sam_type](checkpoint=sam_ckpt).to(device=device)
        sam.eval()
    print("======> Done" )   

    print("======> Load Grounding Detector" )
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print("======> Done" )

    print("======> Load ResNet50")
    # Load ResNet50 model pretrained on ImageNet and remove the top classification layer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resnet50 = models.resnet50(pretrained=True)
    resnet50.fc = nn.Identity()  
    resnet50 = resnet50.to(device)
    resnet50.eval()  

    preprocess_resnet50 = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize
    ])
    print("======> Done" )

    print("======> Load Object Counter" )
    if args.fsoc == 'countr':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_countr = models_mae_cross.__dict__['mae_vit_base_patch16'](norm_pix_loss='store_true')
        model_countr.to(device)
        model_without_ddp = model_countr

        checkpoint = torch.load('./CounTR/output_allnew_dir/FSC147.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=False)
        

        model_countr.eval()
        counter_model = model_countr
    elif args.fsoc == 'DSALVANet':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        parser_input = argparse.ArgumentParser(description="Test code of DSALVANet")
        parser_input.add_argument("-w", "--weight", type=str, default="./DSALVANet/checkpoints/checkpoint_200.pth", help="Path of weight.")
        parser_input.add_argument('--visualize', type=bool, default= False)
        parser_input.add_argument('--fsoc', type=str, default='DSALVANet')

        args_counter = parser_input.parse_args()
        weight_path = args_counter.weight
        counter_model = build_model(weight_path,device)
    print("======> Done" )

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Images/'
    output_path = './outputs/' + args.outdir


    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        infer_time = 0
        if ".DS" not in obj_name:
            persense(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, counter_model, infer_time, resnet50, preprocess_resnet50)



def persense(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, counter_model, infer_time, resnet50, preprocess_resnet50):
    
    print("\n------------> Segment " + obj_name)
    
    ref_image_path = os.path.join(images_path, obj_name, args.ref_idx + '.jpg')
    ref_mask_path = os.path.join(masks_path, obj_name, args.ref_idx + '.png')
    test_images_path = os.path.join(images_path, obj_name)
    

    output_path = os.path.join(output_path, obj_name)
    os.makedirs(output_path, exist_ok=True)

    ref_image = cv2.imread(ref_image_path)
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()


   
    print("======> Getting Class Label" )
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


    supp_mask = Image.open(ref_image_path).convert("RGB")
    supp_image = Image.open(ref_mask_path).convert("RGB")
    raw_image = ImageChops.multiply(supp_image, supp_mask)
    raw_image.save("./ref_images/masked_img.png", "PNG")

    image_file = "./ref_images/masked_img.png"

    model_path = "mucai/vip-llava-7b"
    prompt = "Name the object in the image?"

    with torch.no_grad():

        args_llava = type('Args', (), {
            "model_path": model_path,
            "model_name": get_model_name_from_path(model_path),
            "query": prompt,
            "image_file": image_file,
            "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
        })()
        model_llava = model_llava.cuda()
        output = eval_model(args_llava, model_llava, llava_tokenizer, llava_image_processor)
        
    words = output.split()
    last_word = words[-1]
    last_word = [last_word.replace(".", "")]
    model_llava.cpu()
    
    
    for name, param in sam.named_parameters():
        param.requires_grad = False
    predictor = SamPredictor(sam)
    

    print("======> Obtain Self Location Prior" )
    # Image features encoding
    ref_mask = predictor.set_image(ref_image, ref_mask)
    ref_feat = predictor.features.squeeze().permute(1, 2, 0)

    ref_mask = F.interpolate(ref_mask, size=ref_feat.shape[0: 2], mode="bilinear")
    ref_mask = ref_mask.squeeze()[0]

    # Target feature extraction
    target_feat = ref_feat[ref_mask > 0]
    target_embedding = target_feat.mean(0).unsqueeze(0)
    target_feat = target_embedding / target_embedding.norm(dim=-1, keepdim=True)
    target_embedding = target_embedding.unsqueeze(0)
   
    print('======> Start Testing')
    loop_over = len(os.listdir(test_images_path)) - 2
    for test_idx in tqdm(range(loop_over//2)):

        image_name = f"{test_idx:02}.png" if test_idx < 10 else f"{test_idx}.png"
        output_file = os.path.join(output_path, image_name)
        if os.path.exists(output_file):
            print(f"Skipping {image_name} as it already exists.")
            continue
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        image_height, image_width, _ = test_image.shape
        image_size = (image_height, image_width)

        # Image feature encoding
        predictor.set_image(test_image)
        test_feat = predictor.features.squeeze()

        # Cosine similarity
        C, h, w = test_feat.shape
        test_feat = test_feat / test_feat.norm(dim=0, keepdim=True)
        test_feat = test_feat.reshape(C, h * w)
        sim = target_feat @ test_feat

        sim = sim.reshape(1, 1, h, w)
        sim = F.interpolate(sim, scale_factor=4, mode="bilinear")
        sim = predictor.model.postprocess_masks(
                        sim,
                        input_size=predictor.input_size,
                        original_size=predictor.original_size).squeeze()
        
        
        print("======> Running PerSense" )


        SOURCE_IMAGE_PATH = test_image_path
        class1 = last_word
        class2 = enhance_class_name(class_names=last_word)
        CLASSES = class1
        
        BOX_TRESHOLD = 0.15
        TEXT_TRESHOLD = 0.10

        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        with torch.no_grad():

            # detect objects
            detections = grounding_dino_model.predict_with_classes(
                image=image,
                classes = CLASSES,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )
        class_conf = detections.confidence
        index_conf = np.argmax(class_conf)
        bbox_coord = detections.xyxy[index_conf]

        top_list = []
        filt_sim, cntr_pt = filtered_similarity(sim, bbox_coord) 
        topk_xy_NA, topk_label = point_selection(filt_sim, topk=1)
        
        top_list.append(cntr_pt)
        topk_xy = top_list[0]
        topk_xy = np.array(top_list)
        sim_tgt = (sim - sim.mean()) / torch.std(sim)
        sim_tgt = F.interpolate(sim_tgt.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim_tgt.sigmoid_().unsqueeze(0).flatten(3)

        CLASSES = class2
        print(CLASSES)

        with torch.no_grad():
            detections2 = grounding_dino_model.predict_with_classes(
                image=image,
                classes = CLASSES,
                box_threshold=BOX_TRESHOLD,
                text_threshold=TEXT_TRESHOLD
            )

        with torch.no_grad():
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy, 
                point_labels=topk_label, 
                multimask_output=True,
                attn_sim=attn_sim,  
                target_embedding=target_embedding 
            )
            best_idx = np.argmax(scores)
        
            masks, scores, logits, _ = predictor.predict(
                point_coords=topk_xy,
                point_labels=topk_label,
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

        y, x = np.nonzero(masks[best_idx])
        x_min = x.min()
        x_max = x.max()
        y_min = y.min()
        y_max = y.max()
        input_box = np.array([x_min, y_min, x_max, y_max])
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            box=input_box[None, :],
            mask_input=logits[best_idx: best_idx + 1, :, :], 
            multimask_output=True)
        best_idx = np.argmax(scores)

        bbox = [[y_min, x_min, y_max, x_max]]
       
        lines = [str(y_min), str(x_min), str(y_max), str(x_max),str(scores[best_idx])]
        with open('./DSALVANet/test_data/bbox.txt', 'w') as f:

            for line in lines:
                f.write(line)
                f.write(' ')
            f.write('\n') 


        parser = argparse.ArgumentParser(description="Test code of DSALVANet")
        parser.add_argument("-i", "--img", type=str, default= test_image_path, help="Path of query image.")
        parser.add_argument("-b", "--boxes", type=str, default="./DSALVANet/test_data/bbox.txt", help="Path of bbox coord txt file ")

        if __name__ == '__main__':
            args_dsalva = parser.parse_args()
            img_path,boxes_path = args_dsalva.img, args_dsalva.boxes
            with open(boxes_path, "r") as f:
                lines = f.readlines()
                ori_boxes = []
                for line in lines:
                    data = line.split()
                    ori_boxes.append(list(map(int,data[0:4])))
            src_img = cv2.imread(img_path)
            query, supports = preprocess(src_img, ori_boxes,device)
            

            if args.fsoc == 'countr':
                with torch.no_grad():
                    samples, boxes,ori_boxes, pos, W, H, new_W, new_H = load_image(test_image_path)
                    samples = samples.unsqueeze(0).to(device, non_blocking=True)
                    boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

                    result, elapsed_time, density_pred = run_one_image(samples, boxes, pos, counter_model, W, H, test_idx, new_W, new_H)
                vis_output, pt_priors, count = hybrid_IDM_countr(src_img, ori_boxes, result, test_idx, density_pred)
            elif args.fsoc == 'DSALVANet':
                with torch.no_grad():
                    output = counter_model(query,supports)
                vis_output, pt_priors, count = hybrid_IDM(src_img,ori_boxes,output, test_idx)

            pt_priors_all = pt_priors
            pt_priors = PPSM(sim, pt_priors, count, detections2)
            if not pt_priors:
                pt_priors = pt_priors_all

        batch_size = 16
        pt_np = [
            pt.cpu().detach().numpy().astype(np.int64)
            for pt in pt_priors
        ]
        N = len(pt_np)
 
        coords_all = np.stack(pt_np, axis=0)  # (N,2)
        coords_all = predictor.transform.apply_coords(
            coords_all,
            predictor.original_size
        )                                       # (N,2) 
        coords_torch = torch.as_tensor(
            coords_all,
            dtype=torch.float,
            device=predictor.device
        ).unsqueeze(1)                         # (N,1,2)

        
        labels_all = np.repeat(topk_label, N, axis=0)  # (N,)
        labels_torch = torch.tensor(
            labels_all,
            dtype=torch.int,
            device=predictor.device
        ).unsqueeze(1)                         # (N,1)

        mask_list_final   = []
        prompt_list_final = []

        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                M   = end - start

                pts_batch = coords_torch[start:end]   # (M,1,2)
                lbs_batch = labels_torch[start:end]   # (M,1)

                masks1, scores1, lowres1, _ = predictor.predict_torch(
                    point_coords     = pts_batch,
                    point_labels     = lbs_batch,
                    multimask_output = True,
                    attn_sim         = attn_sim,
                    target_embedding = target_embedding,
                    return_logits    = False
                )
                best1  = scores1.argmax(dim=1)                          # (M,)
                mask_in = lowres1[torch.arange(M), best1].unsqueeze(1)   # (M,1,h,w)

                masks2, scores2, _, _ = predictor.predict_torch(
                    point_coords     = pts_batch,
                    point_labels     = lbs_batch,
                    mask_input       = mask_in,
                    multimask_output = True,
                    return_logits    = False
                )

                batch_lines = []
                for i in range(M):
                   
                    best2 = scores2[i].argmax()             
                    m = masks2[i, best2].cpu().numpy().astype(np.uint8)
                    y, x = np.nonzero(m)
                    try:
                        if x.size == 0:
                            raise ValueError("Empty mask: cannot compute bbox.")
                        x_min, x_max = x.min(), x.max()
                        y_min, y_max = y.min(), y.max()
                    except ValueError as e:
                        print(f"An error occurred with {test_idx}: {e}")
                        continue

                    score_str = str(float(scores2[i][best2]))
                    line = f"{y_min} {x_min} {y_max} {x_max} {score_str}\n"
                    batch_lines.append(line)
                    
                    
                with open('./DSALVANet/test_data/bbox.txt', 'a') as f:
                    f.writelines(batch_lines)
                
            
            with open('./DSALVANet/test_data/bbox.txt', 'r') as file:
                lines = file.readlines()

            if len(lines) > 3:
                with open('./DSALVANet/test_data/bbox.txt', 'w') as file:
                    file.writelines(lines[1:])

                diversity_aware_exemplar_selection(image_size, test_idx, bbox, test_image_path, resnet50, preprocess_resnet50, class1)

        
            if __name__ == '__main__':
                args_dsalva = parser.parse_args()
                img_path,boxes_path = args_dsalva.img, args_dsalva.boxes
                with open(boxes_path, "r") as f:
                    lines = f.readlines()
                    ori_boxes = []
                    for line in lines:
                        data = line.split()
                        ori_boxes.append(list(map(int,data[0:4])))
                src_img = cv2.imread(img_path)
                query, supports = preprocess(src_img, ori_boxes,device)
                

                if args.fsoc == 'countr':
                    samples, boxes,ori_boxes, pos, W, H, new_W, new_H = load_image(test_image_path)
                    samples = samples.unsqueeze(0).to(device, non_blocking=True)
                    boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

                    result, elapsed_time, density_pred = run_one_image(samples, boxes, pos, counter_model, W, H, test_idx, new_W, new_H)
                    vis_output, pt_priors, count = hybrid_IDM_countr(src_img, ori_boxes, result, test_idx, density_pred)
                elif args.fsoc == 'DSALVANet':
                    output = counter_model(query,supports)
                    vis_output, pt_priors, count = hybrid_IDM(src_img,ori_boxes,output, test_idx)

                pt_priors_all = pt_priors

                pt_priors = PPSM(sim, pt_priors, count, detections2)
                if not pt_priors:
                    pt_priors = pt_priors_all
           
            
            pt_np = [pt.cpu().detach().numpy().astype(np.int64) for pt in pt_priors]
            pt_np_arr = np.stack(pt_np, axis=0)             # (N,2)
            N = len(pt_np_arr)

            
            coords = predictor.transform.apply_coords(
                pt_np_arr,
                predictor.original_size
            ) 

            coords_torch = torch.tensor(
                coords,
                dtype=torch.float,
                device=predictor.device
            ).unsqueeze(1)  

            labels_all = np.repeat(topk_label, N, axis=0)    # (N,)
            labels_torch = torch.tensor(
                labels_all,
                dtype=torch.int,
                device=predictor.device
            ).unsqueeze(1)  # → (N,1)

            mask_list_final   = []
            prompt_list_final = []

        
            for start in range(0, N, batch_size):
                end = min(start + batch_size, N)
                M   = end - start

                pts_batch = coords_torch[start:end]   # (M,1,2)
                lbs_batch = labels_torch[start:end]   # (M,1)

                masks_b, scores_b, lowres_b, _ = predictor.predict_torch(
                    point_coords     = pts_batch,
                    point_labels     = lbs_batch,
                    multimask_output = True,
                )
               
                best_idx = scores_b.argmax(dim=1)                # (M,)
                sel_masks = masks_b[torch.arange(M), best_idx]   # (M,H,W)
                sel_masks = sel_masks.cpu().numpy()              # float in [0,1]

                for i in range(M):
                    mask_list_final.append(sel_masks[i])
                    prompt_list_final.append(pt_np_arr[start + i : start + i + 1])

                del masks_b, scores_b, lowres_b, sel_masks
                torch.cuda.empty_cache()
            del masks1, scores1, lowres1, masks2, scores2
            torch.cuda.empty_cache()

            if len(mask_list_final) == 1:
                single = (mask_list_final[0].astype(np.uint8) * 255)
                mask_output_path = os.path.join(output_path, f"{test_idx}.png")
                cv2.imwrite(mask_output_path, single)
            else:
                filtered_masks, filtered_prompts = IMRM(
                    mask_list_final,
                    prompt_list_final,
                    detections
                 )

                if args.visualize:
                    cv2_visualization(
                        test_image,
                        filtered_masks,
                        filtered_prompts,
                        topk_label,
                        output_path,
                        test_idx
                    )

                composite = np.zeros_like(filtered_masks[0], dtype=np.uint8)
                for m in filtered_masks:
                    composite |= m.astype(np.uint8)
                composite *= 255

                mask_output_path = os.path.join(output_path, f"{test_idx}.png")
                cv2.imwrite(mask_output_path, composite)




class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def compute_iou(boxA, boxB):
    yA1, xA1, yA2, xA2 = boxA
    yB1, xB1, yB2, xB2 = boxB

    inter_y1 = max(yA1, yB1)
    inter_x1 = max(xA1, xB1)
    inter_y2 = min(yA2, yB2)
    inter_x2 = min(xA2, xB2)

    inter_area = max(0, inter_y2 - inter_y1) * max(0, inter_x2 - inter_x1)
    areaA = (yA2 - yA1) * (xA2 - xA1)
    areaB = (yB2 - yB1) * (xB2 - xB1)

    union_area = areaA + areaB - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def box_contains(boxA, boxB):
    yA1, xA1, yA2, xA2 = boxA
    yB1, xB1, yB2, xB2 = boxB
    return yA1 <= yB1 and xA1 <= xB1 and yA2 >= yB2 and xA2 >= xB2

def get_bbox_from_mask(mask):
    ys, xs = np.nonzero(mask)
    return [ys.min(), xs.min(), ys.max(), xs.max()]

def IMRM(mask_list, prompt_list, dino_detections, n_clusters=2):
  
    areas = np.array([m.sum() for m in mask_list], dtype=np.float32).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(areas)
    labels = kmeans.labels_


    unique, counts = np.unique(labels, return_counts=True)
    majority = unique[np.argmax(counts)]
    maj_areas = areas[labels == majority].flatten()
    mu_maj = maj_areas.mean()
    sigma_maj = maj_areas.std()


    q1, q3 = np.percentile(areas.flatten(), [25, 75])
    iqr = q3 - q1
    thresh_iqr = q3 + 2 * iqr


    area_thresh = (mu_maj + 2 * sigma_maj) + thresh_iqr

    dino_boxes = []
    for box in dino_detections.xyxy:
        x0, y0, x1, y1 = box.tolist()
        dino_boxes.append([int(y0), int(x0), int(y1), int(x1)])

    independent_boxes = []
    if len(dino_boxes) > 1:
        for i, box_i in enumerate(dino_boxes):
            contains_other = False
            for j, box_j in enumerate(dino_boxes):
                if i != j and box_contains(box_i, box_j):
                    contains_other = True
                    break
            if not contains_other:
                independent_boxes.append(box_i)

    filtered_masks = []
    filtered_prompts = []

    for m, p, a in zip(mask_list, prompt_list, areas.flatten()):
        if a <= area_thresh:
            filtered_masks.append(m)
            filtered_prompts.append(p)
        else:
            if independent_boxes:
                mask_box = get_bbox_from_mask(m)
                iou_scores = [compute_iou(mask_box, db) for db in independent_boxes]
                if max(iou_scores, default=0.0) >= 0.8:
                    filtered_masks.append(m)
                    filtered_prompts.append(p)
            pass

    return filtered_masks, filtered_prompts



def filtered_similarity(sim_matrix, bbox):
    cimg = np.zeros_like(sim_matrix.cpu(), np.uint8)
    bbox = np.int32(bbox)
    cimg_box = cv2.rectangle(cimg,(bbox[0],bbox[1]), (bbox[2], bbox[3]) , 255 , -1)
    center_pt = ((bbox[0]+ bbox[2])//2, (bbox[1]+bbox[3])//2)
    center_pt = np.int32(center_pt)
    radius = 2
    cv2.circle(cimg_box, center_pt, radius, (255, 255, 0), 2)
    sim_mat_new = np.multiply(sim_matrix.cpu(), cimg_box)

    return sim_mat_new, center_pt

def point_selection(mask_sim, topk=1):
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label

def PPSM(similarity, point_priors, cnt, dino_boxes):
    final_pts = []
    for pnt in point_priors:
        x, y = int(pnt[0]), int(pnt[1])
        for (x0, y0, x1, y1) in dino_boxes.xyxy:
            if x0 <= x <= x1 and y0 <= y <= y1:
                final_pts.append(pnt)
                break
    return final_pts


def enhance_class_name(class_names: List[str]) -> List[str]:
    vowels = {'a', 'i', 'o', 'u'}
    enhanced_names = []
    for class_name in class_names:
        if class_name[-1].lower() == 's':
            enhanced_names.append(f"all {class_name}")
        elif class_name[-1].lower() == 'y':
            if len(class_name) > 1 and class_name[-2].lower() == 'e':
                enhanced_names.append(f"all {class_name}s")
            else:
                enhanced_names.append(f"all {class_name[:-1]}ies")
        elif class_name[-1].lower() == 'h':
            enhanced_names.append(f"all {class_name}es")
        elif class_name[-1].lower() in vowels:
            enhanced_names.append(f"all {class_name}es")
        else:
            enhanced_names.append(f"all {class_name}s")
    return enhanced_names




def parse_bounding_boxes(file_path):
    bounding_boxes = []
    with open(file_path, 'r') as f:
        for line in f:
            coords = list(map(float, line.strip().split()))
            bbox = {
                'coords': coords[:4],  # y_min, x_min, y_max, x_max
                'score': coords[4]  # SAM score
            }
            bounding_boxes.append(bbox)
    return bounding_boxes

def calculate_area(box):
    y_min, x_min, y_max, x_max= box
    return (x_max - x_min) * (y_max - y_min)

def extract_visual_features(
    bounding_boxes,      
    test_image_path: str,
    resnet50,            # pretrained ResNet50 model
    preprocess_resnet50  
) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = cv2.imread(test_image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]
    inp = preprocess_resnet50(image).unsqueeze(0).to(device)  # [1,3,Ht,Wt]
    _, _, Ht, Wt = inp.shape

    backbone = torch.nn.Sequential(*list(resnet50.children())[:-2]).to(device).eval()
    with torch.no_grad():
        feat_map = backbone(inp)                              # [1,C,Hf,Wf]
    _, C, Hf, Wf = feat_map.shape

    spatial_scale = Hf / Ht

    rois = []
    for bbox in bounding_boxes:
        if isinstance(bbox, list) and len(bbox) == 4:
            y0, x0, y1, x1 = bbox
        else:
            y0, x0, y1, x1 = [int(v) for v in bbox["coords"]]
        x1s = x0 * (Wt / orig_w)
        y1s = y0 * (Ht / orig_h)
        x2s = x1 * (Wt / orig_w)
        y2s = y1 * (Ht / orig_h)
        rois.append([0, x1s, y1s, x2s, y2s])
    rois = torch.tensor(rois, device=device, dtype=torch.float32)  # [N,5]

    pooled = roi_align(
        feat_map,
        rois,
        output_size=(3, 3),
        spatial_scale=spatial_scale,
        sampling_ratio=2
    )  

    feats = pooled.mean(dim=[2, 3])
    return feats.cpu().numpy()


def select_highest_scorers(regions):
    filtered_regions = {}
    total_boxes = 0  
    
    for region, boxes in regions.items():
        high_scorers = [box for box in boxes if box['score'] > 0.8]
        total_boxes += len(high_scorers) 
        
        if high_scorers:
            filtered_regions[region] = high_scorers

    if total_boxes < 3:
        filtered_regions = regions
    
    return filtered_regions


def calculate_intersection_over_union(boxA, boxB):
    xA = max(boxA[1], boxB[1])
    yA = max(boxA[0], boxB[0])
    xB = min(boxA[3], boxB[3])
    yB = min(boxA[2], boxB[2])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = calculate_area(boxA)
    boxBArea = calculate_area(boxB)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou, boxAArea, boxBArea


def normalize_values(values):
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val - min_val == 0:
        return np.zeros_like(values)
    return (values - min_val) / (max_val - min_val)


def cluster_by_visual_similarity(bounding_boxes, image_path, resnet50, preprocess_resnet50, ref_boxes, filename, n_clusters=3, min_cluster_size=3):
    regions = {}
    similarities = {}
    filtered_boxes = [bbox for bbox in bounding_boxes if bbox['score'] > 0]

    if not filtered_boxes:
        filtered_boxes = bounding_boxes

    areas = []
    aspect_ratios = []

    for bbox in filtered_boxes:
        x1, y1, x2, y2 = bbox['coords']
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = max(width, height) / min(width, height)  # Ensure large_side / small_side
        
        areas.append(area)
        aspect_ratios.append(aspect_ratio)

    mean_area = np.mean(areas) if areas else 0
    mean_aspect_ratio = np.mean(aspect_ratios) if aspect_ratios else 1  # Avoid division by zero

    seg_fault = False

    if len(filtered_boxes) >= 3:
        
        features = extract_visual_features(filtered_boxes, image_path, resnet50, preprocess_resnet50)
        scaler = StandardScaler()
        features = scaler.fit_transform(features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)  # Use a fixed random state for reproducibility
        labels = kmeans.fit_predict(features)
        cluster_centers = kmeans.cluster_centers_

        all_similarities = []

        for label in np.unique(labels):
            cluster_boxes = [box for i, box in enumerate(filtered_boxes) if labels[i] == label]
            cluster_features = np.array([features[i] for i, _ in enumerate(filtered_boxes) if labels[i] == label])

            center = cluster_centers[label].reshape(1, -1)
            cluster_similarities = cosine_similarity(cluster_features, center).flatten()  # Cosine similarity output
            cluster_similarities = (cluster_similarities + 1) / 2  # Normalize to [0, 1]
            all_similarities.extend(cluster_similarities)
            regions[label] = cluster_boxes
            similarities[label] = cluster_similarities
    else:
        print(f"Only {len(filtered_boxes)}: segmentation fault.")
        seg_fault = True
        regions = {0: filtered_boxes} 
        similarities[0] = [1 for _ in filtered_boxes] 

    return regions,similarities, seg_fault, mean_area, mean_aspect_ratio


def categorize_by_area_distribution(selected_boxes):
    areas = np.array([calculate_area(bbox['coords']) for bbox in selected_boxes])
    percentiles = np.percentile(areas, [33, 66])  # 33rd and 66th percentiles

    size_categories = {'small': [], 'medium': [], 'large': []}

    for bbox, area in zip(selected_boxes, areas):
        if area <= percentiles[0]:
            size_categories['small'].append(bbox)
        elif area <= percentiles[1]:
            size_categories['medium'].append(bbox)
        else:
            size_categories['large'].append(bbox)

    return size_categories

def select_exemplars_by_size(clusters, ref_boxes, similarities, filename, image_path, resnet50, preprocess_resnet50, class1, mean_area, mean_aspect_ratio):
    final_selection = []

    ref_features = []
    ref_aspect_ratios = []
    
    for bbox in ref_boxes:
        if isinstance(bbox, list) and len(bbox) == 4:
            y_min, x_min, y_max, x_max = bbox
        else:
            y_min, x_min, y_max, x_max = [int(coord) for coord in bbox['coords']]
        
        ref_box_coords = [y_min, x_min, y_max, x_max]
        ref_feature = extract_visual_features([{'coords': ref_box_coords}], image_path, resnet50, preprocess_resnet50)
        ref_features.append(ref_feature[0])
        
        height = y_max - y_min
        width = x_max - x_min
        aspect_ratio = height / width if width > 0 else 1  # Prevent division by zero
        ref_aspect_ratios.append(aspect_ratio)
    
    mean_ref_vector = np.mean(ref_features, axis=0)
    mean_ref_aspect_ratio = np.mean(ref_aspect_ratios)

    reference_areas = []
    for bbox in ref_boxes:
        if isinstance(bbox, list) and len(bbox) == 4:
            y_min, x_min, y_max, x_max = bbox
        else:
            y_min, x_min, y_max, x_max = [int(coord) for coord in bbox['coords']]
        reference_areas.append(calculate_area([y_min, x_min, y_max, x_max]))

    mean_ref_area = np.mean(reference_areas)
    all_closeness_to_mean = []
    all_aspect_ratio_deviations = []

    bbox_scores = []
    norm_idx = 0

    for region, boxes in clusters.items():
        cluster_features = extract_visual_features(boxes, image_path, resnet50, preprocess_resnet50)
        
        for idx, box in enumerate(boxes):
            bbox_area = calculate_area(box['coords'])
            closeness_to_mean = abs(bbox_area - mean_area)
            all_closeness_to_mean.append(closeness_to_mean)

            y_min, x_min, y_max, x_max = [int(coord) for coord in box['coords']]
            height = y_max - y_min
            width = x_max - x_min
            aspect_ratio = max(width, height) / min(width, height)
            aspect_ratio_deviation = abs(aspect_ratio - mean_aspect_ratio)
            all_aspect_ratio_deviations.append(aspect_ratio_deviation)
            
            box_feature_vector = cluster_features[idx].reshape(1, -1)
            sim_with_ref = cosine_similarity(box_feature_vector, mean_ref_vector.reshape(1, -1)).flatten()[0]
            if sim_with_ref < 0:
                sim_with_ref = (sim_with_ref + 1) / 2

            # Weighted scoring
            log_dist = np.log(closeness_to_mean)
            weighted_score = (
                -0.25 * log_dist +  
                0.25 * similarities[region][idx] +  
                0.25 * sim_with_ref + 
                -0.25 * aspect_ratio_deviation  
            )

            bbox_scores.append({
                'coords': box['coords'],
                'score': weighted_score,
                'area': bbox_area 
            })
            norm_idx += 1
    

    size_categories = categorize_by_area_distribution(bbox_scores)
    final_selection = []
    
    for size in ['small', 'medium', 'large']:
        if size_categories[size]:
            sorted_boxes = sorted(size_categories[size], key=lambda x: -x['score'])
            top_box = sorted_boxes[0]  # Select the top box from this size category
            final_selection.append(top_box)
    final_selection = remove_overlapping_boxes(final_selection, bbox_scores)

    return final_selection


def remove_overlapping_boxes(final_selection, global_region):
    start_idx = len(final_selection)
    
    while True:
        to_remove = []
        for i in range(len(final_selection)):
            for j in range(i + 1, len(final_selection)):
                iou, area_i, area_j = calculate_intersection_over_union(final_selection[i]['coords'], final_selection[j]['coords'])
                if iou > 0.5:
                    if area_i > area_j:
                        to_remove.append(j)
                    else:
                        to_remove.append(i)
        new_selection = [box for idx, box in enumerate(final_selection) if idx not in to_remove]

        if len(new_selection) == len(final_selection):
            break
        
        final_selection = new_selection
        idx = start_idx  
        while len(final_selection) < 3 and idx < len(global_region):
            if global_region[idx] not in final_selection:
                final_selection.append(global_region[idx])
            idx += 1
        start_idx = idx

    return final_selection


def diversity_aware_exemplar_selection(image_shape, filename, bboxes, test_image_path, resnet50, preprocess_resnet50, class1):
    selected_boxes = parse_bounding_boxes('./DSALVANet/test_data/bbox.txt')
    
    regions, distances, seg_fault, mean_area, mean_aspect_ratio = cluster_by_visual_similarity(selected_boxes, test_image_path, resnet50, preprocess_resnet50, bboxes, filename)
    filtered_regions = select_highest_scorers(regions)
    exemplars = select_exemplars_by_size(filtered_regions, bboxes, distances, filename, test_image_path, resnet50, preprocess_resnet50, class1, mean_area, mean_aspect_ratio)
    with open('./DSALVANet/test_data/bbox.txt', 'w') as file:
        for ex in exemplars:
            line = f"{int(ex['coords'][0])} {int(ex['coords'][1])} {int(ex['coords'][2])} {int(ex['coords'][3])} {ex['score']}\n"
            file.write(line)

    return seg_fault

if __name__ == '__main__':
    main()

