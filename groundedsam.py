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
# warnings.filterwarnings('ignore')

from show import *
from per_segment_anything import sam_model_registry, SamPredictor

from ViPLLaVA.llava.model.builder import load_pretrained_model
from ViPLLaVA.llava.mm_utils import get_model_name_from_path
from ViPLLaVA.llava.eval.run_llava import eval_model

from PIL import Image , ImageChops
from torchvision import transforms

from GroundingDINO.groundingdino.util.inference import Model
from typing import List
import supervision as sv
from segment_anything import SamPredictor
import gc
import time




def get_arguments():
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', type=str, default='./data')
    parser.add_argument('--outdir', type=str, default='baseline')
    parser.add_argument('--ckpt', type=str, default='./sam_vit_h_4b8939.pth')
    parser.add_argument('--sam_type', type=str, default='vit_h')
    parser.add_argument('--ref_idx', type=str, default='00')
    parser.add_argument('--visualize', type=bool, default= False) # Change to True for visualization
    
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

    print("======> Load GroundingDINO" )
    GROUNDING_DINO_CONFIG_PATH = "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    GROUNDING_DINO_CHECKPOINT_PATH = "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    grounding_dino_model = Model(model_config_path=GROUNDING_DINO_CONFIG_PATH, model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH)
    print("======> Done" )

    images_path = args.data + '/Images/'
    masks_path = args.data + '/Images/'
    output_path = './outputs/' + args.outdir

    if not os.path.exists('./outputs/'):
        os.mkdir('./outputs/')
    
    for obj_name in os.listdir(images_path):
        infer_time = 0
        if ".DS" not in obj_name:
            persam_f(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, infer_time)


def persam_f(args, obj_name, images_path, masks_path, output_path, llava_tokenizer, model_llava, llava_image_processor, sam, grounding_dino_model, infer_time):
    
    print("\n------------> Segment " + obj_name)
    avg_iter = 0
  
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
    raw_image.save("/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/masked_img.png", "PNG")

    image_file = "/l/users/muhammad.siddiqui/Personalize-SAM/Example_images/masked_img.png"

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
    



    print('======> Applying Grounded SAM')
    loop_over = len(os.listdir(test_images_path))
    for test_idx in tqdm(range(loop_over//2)):
        # Load test image
        test_idx = '%02d' % test_idx
        test_image_path = test_images_path + '/' + test_idx + '.jpg'
        test_image = cv2.imread(test_image_path)
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)

        # Image feature encoding
        predictor.set_image(test_image)
        
        
        print("======> Running GroundingDINO Baseline" )

        SOURCE_IMAGE_PATH = test_image_path
        CLASSES = last_word
        print(CLASSES)
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

        detections.mask = segment(
            sam_predictor=predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy
        )

      
        # annotate image with detections
        box_annotator = sv.BoxAnnotator()
        mask_annotator = sv.MaskAnnotator()
        mask_annotator.opacity = 0.8

        # Visualization
        if args.visualize:
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            plt.figure(figsize=(10, 10))
            plt.axis('off')
            annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
            plt.imshow(annotated_image)
        
            vis_mask_output_path = os.path.join(output_path, f'vis_mask_{test_idx}.jpg')
            with open(vis_mask_output_path, 'wb') as outfile:
                plt.savefig(outfile, format='jpg')
            plt.close()


        composite_mask = np.zeros_like(detections.mask[0], dtype=np.uint8)  # Initialize composite mask
        for mask in detections.mask:
            composite_mask |= mask.astype(np.uint8) * 255  # Combine all masks using logical OR operation
        mask_output_path = os.path.join(output_path, f'{test_idx}.png')
        cv2.imwrite(mask_output_path, composite_mask)
        # plt.imshow(composite_mask, cmap='gray')  # Plot the composite mask
        # release_memory(test_image)

        


def release_memory(test_image):
    # Clear variables
    del test_image
    # Explicitly release GPU memory
    torch.cuda.empty_cache()
    # Perform garbage collection
    gc.collect()


def segment(sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, logits = sam_predictor.predict(
            box=box,
            multimask_output=True
        )
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)

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
