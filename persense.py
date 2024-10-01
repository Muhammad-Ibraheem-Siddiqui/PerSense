# Official Implementation of PerSense

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import os
import cv2
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor

from DSALVANet.utils.PerSense_modules import IDM
from DSALVANet.utils.data_preprocess import preprocess
from DSALVANet.utils.model_helper import build_model
from DSALVANet.utils.PerSense_countr import IDM_countr

from ViPLLaVA.llava.model.builder import load_pretrained_model
from ViPLLaVA.llava.mm_utils import get_model_name_from_path
from ViPLLaVA.llava.eval.run_llava import eval_model

from PIL import Image , ImageChops
from torchvision import transforms

from GroundingDINO.groundingdino.util.inference import Model
from typing import List
import supervision as sv
import gc
import time
from CounTR import models_mae_cross
from CounTR.demo import load_image, run_one_image



def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='PerSense')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--visualize', type=bool, default= False) # Change to True for visualization
    parser.add_argument('--fsoc', type=str, default='DSALVANet') #use countr for COUNTR BMVC 22

    
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
        parser_input.add_argument('--fsoc', type=str, default='DSALVANet') #use countr for COUNTR BMVC 22

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
            persense(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, counter_model, infer_time)



def persense(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, counter_model, infer_time):

    obj_count = 0
    avg_iter = 0
    
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

    ref_mask = cv2.imread(ref_mask_path)
    ref_mask = cv2.cvtColor(ref_mask, cv2.COLOR_BGR2RGB)

    gt_mask = torch.tensor(ref_mask)[:, :, 0] > 0 
    gt_mask = gt_mask.float().unsqueeze(0).flatten(1).cuda()


   
    print("======> Getting Class Label using LLM" )
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


    supp_mask = Image.open(ref_image_path).convert("RGB")
    supp_image = Image.open(ref_mask_path).convert("RGB")
    raw_image = ImageChops.multiply(supp_image, supp_mask)
    raw_image.save("./ref_images/masked_img.png", "PNG")

    image_file = "./ref_images/masked_img.png"

    model_path = "mucai/vip-llava-7b"
    prompt = "Name the object in the image?"

    args_llava = type('Args', (), {
        "model_path": model_path,
        "model_name": get_model_name_from_path(model_path),
        "query": prompt,
        "image_file": image_file,
        "conv_mode": None, "model_base": None, "temperature": 0.2, "top_p": None, "num_beams": 1, "max_new_tokens": 512, "sep": ",",
    })()

    output = eval_model(args_llava, model_llava, llava_tokenizer, llava_image_processor)
    print(output)
    words = output.split()
    last_word = words[-1]
    last_word = [last_word.replace(".", "")]
    print (last_word)
    
    
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
    loop_over = len(os.listdir(test_images_path))
    for test_idx in tqdm(range(loop_over//2)):

        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

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
        # CLASSES = [class1[0], class2[0]]
        CLASSES = class1
        
        BOX_TRESHOLD = 0.15
        TEXT_TRESHOLD = 0.10

        # load image
        image = cv2.imread(SOURCE_IMAGE_PATH)

        # detect objects
        detections = grounding_dino_model.predict_with_classes(
            image=image,
            # classes=enhance_class_name(class_names=CLASSES),
            classes = CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )
        
        # annotate image with detections
        class_conf = detections.confidence
        index_conf = np.argmax(class_conf)
        bbox_coord = detections.xyxy[index_conf]


        # Positive location prior
        top_list = []
        filt_sim, cntr_pt = filtered_similarity(sim, bbox_coord) #Filtering the values of cosine similarity falling under BBox with maximum conf value
        topk_xy_NA, topk_label = point_selection(filt_sim, topk=1)
        
        top_list.append(cntr_pt)
        topk_xy = top_list[0]
        topk_xy = np.array(top_list)

        # Obtain the target guidance for cross-attention layers
        sim_tgt = (sim - sim.mean()) / torch.std(sim)
        sim_tgt = F.interpolate(sim_tgt.unsqueeze(0).unsqueeze(0), size=(64, 64), mode="bilinear")
        attn_sim = sim_tgt.sigmoid_().unsqueeze(0).flatten(3)

        #running DINO for all objects class to discard points outside the BBox
        CLASSES = class2
        print(CLASSES)

        # detect objects
        detections2 = grounding_dino_model.predict_with_classes(
            image=image,
            # classes=enhance_class_name(class_names=CLASSES),
            classes = CLASSES,
            box_threshold=BOX_TRESHOLD,
            text_threshold=TEXT_TRESHOLD
        )


        # First-step prediction
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy, 
            point_labels=topk_label, 
            multimask_output=True,
            attn_sim=attn_sim,  # Target-guided Attention
            target_embedding=target_embedding  # Target-semantic Prompting
        )
        best_idx = np.argmax(scores)


        # Cascaded Post-refinement-1
       
        masks, scores, logits, _ = predictor.predict(
            point_coords=topk_xy,
            point_labels=topk_label,
            mask_input=logits[best_idx: best_idx + 1, :, :],
            multimask_output=True)
        best_idx = np.argmax(scores)


        # Cascaded Post-refinement-2
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
       
        lines = [str(y_min), str(x_min), str(y_max), str(x_max),str(scores[best_idx])]
        with open('./DSALVANet/test_data/bbox.txt', 'w') as f:

            for line in lines:
                f.write(line)
                f.write(' ') 

        parser = argparse.ArgumentParser(description="Test code of DSALVANet")
        # parser.add_argument("-w", "--weight", type=str, default="/home/muhammad.siddiqui/Desktop/muhammad.siddiqui/Personalize-SAM/DSALVANet/checkpoints/checkpoint_200.pth", help="Path of weight.")
        parser.add_argument("-i", "--img", type=str, default= test_image_path, help="Path of query image.")
        parser.add_argument("-b", "--boxes", type=str, default="./DSALVANet/test_data/bbox.txt", help="Path of bbox coord txt file ")
        parser.add_argument('--visualize', type=bool, default= False)
        parser.add_argument('--fsoc', type=str, default='DSALVANet') #use countr for COUNTR BMVC 22

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

                # Test on the new image
                samples, boxes,ori_boxes, pos, W, H, new_W, new_H = load_image(test_image_path)
                samples = samples.unsqueeze(0).to(device, non_blocking=True)
                boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

                result, elapsed_time, density_pred = run_one_image(samples, boxes, pos, counter_model, W, H, test_idx, new_W, new_H)
                vis_output, pt_priors, count = IDM_countr(src_img, ori_boxes, result, test_idx, density_pred)
            elif args.fsoc == 'DSALVANet':
                output = counter_model(query,supports)
                vis_output, pt_priors, count = IDM(src_img,ori_boxes,output, test_idx)

            max_conf_pt = topk_xy[0] # including the max conf point in the prompt list
            pt_priors = point_prompt_select(sim, pt_priors, count, detections2) # to compare possible points with similrity map for filtering the accurate ones

            # pt_priors = pt_priors.cpu().detach().numpy().astype(np.int64)
            # set_path = './outputs/' + 'counter_output'
            # if not os.path.exists(set_path):
            #     os.mkdir('./outputs/counter_output')
            # counter_output_path = os.path.join(set_path, f'{test_idx}.png')
            # cv2.imwrite(counter_output_path,vis_output)
            # print('Counting Finish.')
        pt_list = []
        
        cnt = 0
        for pt in pt_priors:
            pt = pt.cpu().detach().numpy().astype(np.int64)
            cnt += 1
            # pt[0], pt[1] = pt[1], pt[0]
            if cnt == 1:
                pt_list.append(pt)
            pt_list[0] = pt
            point = np.array(pt_list)


            masks, scores, logits, _ = predictor.predict(
                point_coords=point, 
                point_labels=topk_label, 
                multimask_output=True,
                attn_sim=attn_sim,  # Target-guided Attention
                target_embedding=target_embedding  # Target-semantic Prompting
            )
            best_idx = np.argmax(scores)

           
            masks, scores, logits, _ = predictor.predict(
                point_coords=point,
                point_labels=topk_label,
                mask_input=logits[best_idx: best_idx + 1, :, :],
                multimask_output=True)
            best_idx = np.argmax(scores)

            # Cascaded Post-refinement-2
            y, x = np.nonzero(masks[best_idx])
            x_min = x.min()
            x_max = x.max()
            y_min = y.min()
            y_max = y.max()
            input_box = np.array([x_min, y_min, x_max, y_max])

            lines = [str(y_min), str(x_min), str(y_max), str(x_max), str(scores[best_idx])] #Including score value for the best mask for each filtered point
            with open('./DSALVANet/test_data/bbox.txt', 'a') as f:
                f.write('\n')
                for line in lines:
                    f.write(line)
                    f.write(' ')   
            
            
        # Read the contents of the text file
        with open('./DSALVANet/test_data/bbox.txt', 'r') as file:
            lines = file.readlines()
        
        # Extract the first line (header) and skip it for sorting
        max_conf_bbox = lines.pop(0)

        # Sort the lines based on the values in the 5th column (index 4)
        sorted_lines = sorted(lines, key=lambda x: float(x.split()[4]), reverse=True)

        # Write the sorted lines into a new text file
        with open('./DSALVANet/test_data/bbox.txt', 'w') as file:
            file.write(max_conf_bbox)  # Write back the header
            file.writelines(sorted_lines)
        
        # Read the contents of the text file
        with open('./DSALVANet/test_data/bbox.txt', 'r') as file:
            lines = file.readlines()

        # Keep only the first n lines
        first_n_lines = lines[:4]

        # Write the first six lines into the text file, overwriting its contents
        with open('./DSALVANet/test_data/bbox.txt', 'w') as file:
            file.writelines(first_n_lines)


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
            # model = build_model(weight_path,device)
            

            if args.fsoc == 'countr':

                # Test on the new image
                samples, boxes,ori_boxes, pos, W, H, new_W, new_H = load_image(test_image_path)
                samples = samples.unsqueeze(0).to(device, non_blocking=True)
                boxes = boxes.unsqueeze(0).to(device, non_blocking=True)

                result, elapsed_time, density_pred = run_one_image(samples, boxes, pos, counter_model, W, H, test_idx, new_W, new_H)
                vis_output, pt_priors, count = IDM_countr(src_img, ori_boxes, result, test_idx, density_pred)
            elif args.fsoc == 'DSALVANet':
                output = counter_model(query,supports)
                vis_output, pt_priors, count = IDM(src_img,ori_boxes,output, test_idx)

            pt_priors_all = pt_priors

            pt_priors = point_prompt_select(sim, pt_priors, count, detections2) # to compare possible points with similrity map for filtering the accurate ones
            if not pt_priors:
                pt_priors = pt_priors_all


            # pt_priors = pt_priors.cpu().detach().numpy().astype(np.int64)
            # set_path = './outputs/' + 'counter_output'
            # if not os.path.exists(set_path):
            #     os.mkdir('./outputs/counter_output')
            # counter_output_path = os.path.join(set_path, f'{test_idx}.png')
            # cv2.imwrite(counter_output_path,vis_output)
            # print('Counting Finish.')
            
        mask_list_final = []
        pt_list_final = []
        prompt_list_final = []
        cnt = 0
        for pt in pt_priors:
            pt = pt.cpu().detach().numpy().astype(np.int64)
            cnt += 1
            
            if cnt == 1:
                pt_list_final.append(pt)
            pt_list_final[0] = pt
            point = np.array(pt_list_final)


            masks, scores, logits, logits_high = predictor.predict(
            point_coords=point,
            point_labels=topk_label,
            multimask_output=True)


            best_idz = np.argmax(scores)
            prompt_list_final.append(point)
            mask_list_final.append(masks[best_idz])
        best_idx = np.argmax(mask_list_final)

        #Change the visualization argument to True to visualize the mask
        if args.visualize:
            visualization(test_image, mask_list_final, prompt_list_final, topk_label, output_path, test_idx)


        composite_mask = np.zeros_like(mask_list_final[0], dtype=np.uint8)  # Initialize composite mask
        for mask in mask_list_final:
            composite_mask |= mask.astype(np.uint8) * 255  # Combine all masks using logical OR operation
        mask_output_path = os.path.join(output_path, f'{test_idx}.png')
        cv2.imwrite(mask_output_path, composite_mask)


        release_memory(test_image, test_feat )


class Mask_Weights(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(2, 1, requires_grad=True) / 3)


def visualization(test_image, mask_list_final, prompt_list_final, topk_label, output_path, test_idx):
        
        fig = plt.figure(figsize=(10, 10))
        plt.imshow(test_image)
        for z in range(len(mask_list_final)):
            show_mask(mask_list_final[z], plt.gca())
            show_points(prompt_list_final[z], topk_label, plt.gca())
        plt.title(f"PerSense Mask", fontsize=18)
        plt.axis('off')
        vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
        with open(vis_mask_output_path, 'wb') as outfile:
            fig.savefig(outfile, format='jpg')
        
        plt.close(fig)


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

# Function to release memory
def release_memory(test_image, test_feat):
    # Clear variables
    del test_image, test_feat
    # Explicitly release GPU memory
    torch.cuda.empty_cache()
    # Perform garbage collection
    gc.collect()

def point_selection(mask_sim, topk=1):
    # Top-1 point selection
    w, h = mask_sim.shape
    topk_xy = mask_sim.flatten(0).topk(topk)[1]
    topk_x = (topk_xy // h).unsqueeze(0)
    topk_y = (topk_xy - topk_x * h)
    topk_xy = torch.cat((topk_y, topk_x), dim=0).permute(1, 0)
    topk_label = np.array([1] * topk)
    topk_xy = topk_xy.cpu().numpy()
    
    return topk_xy, topk_label

def point_prompt_select(similarity, point_priors, cnt, dino_boxes):
    max_conf = torch.tensor(similarity.max()),
    # b = similarity.min()

    final_pts = []
    for pnt in point_priors:
        check = similarity [pnt[1], pnt[0]] # coords inverted because it is coming in this format from the counter
        if cnt == 1:
            a = torch.sqrt(torch.tensor([4]))
        else:
            a = torch.div(torch.tensor([cnt]), torch.sqrt(torch.tensor([2])))
        threshold = torch.div(max_conf[0],a.cuda())
        verified_point = False
        for boxes in dino_boxes.xyxy:
            if (check >= threshold) and (boxes[0] <= pnt[0] <= boxes[2]) and (boxes[1] <= pnt[1] <= boxes[3]): # To check if the point lies within the bounding boxes predicted by DINO
                verified_point = True
                break
        if verified_point:    
            final_pts.append(pnt)

    return final_pts

def enhance_class_name(class_names: List[str]) -> List[str]:
    vowels = {'a', 'i', 'o', 'u'}
    enhanced_names = []
    for class_name in class_names:
        # Check if the word ends with 's' or 'y'
        if class_name[-1].lower() == 's':
            enhanced_names.append(f"all {class_name}")
        elif class_name[-1].lower() == 'y':
            if len(class_name) > 1 and class_name[-2].lower() == 'e':
                enhanced_names.append(f"all {class_name}s")
            else:
                enhanced_names.append(f"all {class_name[:-1]}ies")
         # Check if the word ends with 'h'
        elif class_name[-1].lower() == 'h':
            enhanced_names.append(f"all {class_name}es")
        # For all other cases
        elif class_name[-1].lower() in vowels:
            enhanced_names.append(f"all {class_name}es")
        else:
            enhanced_names.append(f"all {class_name}s")
    return enhanced_names



if __name__ == '__main__':
    main()
