import cv2
import numpy as np
import torch
import matplotlib.pyplot  as plt
import math
import os
from sys import argv
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed

def apply_scoremap(image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=np.float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def IDM_countr(src_img, boxes, pre_cnt, test_idx, density_pred):

        gray = cv2.cvtColor((density_pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

        # threshold to binary
        thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)[1]

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        contour_img = density_pred.copy()
        contours = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        contour_areas = [cv2.contourArea(contour) for contour in contours]

        # Calculate mean and standard deviation of contour areas
        mean_area = np.mean(contour_areas)
        std_area = np.std(contour_areas)

        threshold_multiplier = 2.0  # Adjust multiplier as needed
        adaptive_threshold = mean_area + threshold_multiplier * std_area
        composite_contours = [contour for contour, area in zip(contours, contour_areas) if area > adaptive_threshold]
        cluster_count = 0
        coord_list = []
        eps = 1e-15

        for large_contour in composite_contours:
               
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [large_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)                
                _, thresholded_dist = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, cv2.THRESH_BINARY)
                thresholded_dist = np.uint8(thresholded_dist)
                child_contours,_ = cv2.findContours(thresholded_dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for child_contour in child_contours:
                        M = cv2.moments(child_contour)
                        cX_child = int(M["m10"] / (M["m00"]+ eps))
                        cY_child = int(M["m01"] / (M["m00"]+ eps))
                        coords_child = torch.tensor([cX_child,cY_child])
                        cv2.drawContours(contour_img, [child_contour], 0, (0,255,0), 2)
                        cv2.circle(contour_img, (cX_child, cY_child), 1, (255, 255, 255), -1)
                        # cluster_count = cluster_count + 1
                        coord_list.append(coords_child)             
               
       
        for cntr in contours:
                M = cv2.moments(cntr)
                cX = int(M["m10"] / (M["m00"]+ eps)) #to avoid division by zero
                cY = int(M["m01"] / (M["m00"]+ eps))
                coords = torch.tensor([cX,cY])
                cv2.drawContours(contour_img, [cntr], 0, (0,255,0), 2)
                cv2.circle(contour_img, (cX, cY), 1, (255, 255, 255), -1)
                cluster_count = cluster_count + 1
                coord_list.append(coords)
                       
        
        # print('Count of Objects',pre_cnt)

        ## save result
        #blob_img_path = "./outputs/" + "contours_img"
        #if not os.path.exists(blob_img_path):
        #        os.mkdir('./outputs/contours_img')
        #blob_img_path = os.path.join(blob_img_path, f'{test_idx}.png')
        #cv2.imwrite(blob_img_path, 255*contour_img)



        output = apply_scoremap(src_img, density_pred)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        for box in boxes:
            y1, x1, y2, x2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(output, "Result:{0}".format(pre_cnt), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
        return output , coord_list, pre_cnt


def hybrid_IDM_countr(src_img, boxes, pre_cnt, test_idx, density_pred):
    """
    Combines contour-based and peak+watershed instance detection.
    Returns:
      - output: visualization image
      - coord_list: list of torch.Tensor([x, y]) union of detected points
      - pre_cnt: predicted count = ceil(sum of density)
    """

    gray = cv2.cvtColor((density_pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_density = gray.copy()
    coord_set = set()
    eps = 1e-15

    _,thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
    eroded = cv2.erode(thresh, kernel1, iterations=1)
    contours,_ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    areas = [cv2.contourArea(c) for c in contours]
    if areas:
        mean_area = np.mean(areas)
        std_area  = np.std(areas)
        adaptive_thresh = mean_area + 2.0 * std_area

        composite_contours = [
            c for c, a in zip(contours, areas) if a > adaptive_thresh
        ]
        for large_contour in composite_contours:
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [large_contour], -1, 255, thickness=cv2.FILLED)
            dist = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
            _, tdist = cv2.threshold(dist, 0.5 * dist.max(), 255, cv2.THRESH_BINARY)
            tdist = tdist.astype(np.uint8)

            child_contours, _ = cv2.findContours(tdist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cc in child_contours:
                M = cv2.moments(cc)
                cX = int(M["m10"] / (M["m00"] + eps))
                cY = int(M["m01"] / (M["m00"] + eps))
                coord_set.add((cX, cY))

  
        for c in contours:
            M = cv2.moments(c)
            cX = int(M["m10"] / (M["m00"] + eps))
            cY = int(M["m01"] / (M["m00"] + eps))
            coord_set.add((cX, cY))

    mean_gray = gray_density.mean()
    std_gray = gray_density.std()
    adaptive_thresh = mean_gray + 2*std_gray
    coords = peak_local_max(gray_density, min_distance=5, threshold_abs=adaptive_thresh)

    for (y, x) in coords:
         coord_set.add((x, y))


    coord_list = [torch.tensor([x, y]) for (x, y) in coord_set]

    # --- Visualization output ---
    output = apply_scoremap(src_img, density_pred)
    output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

    # draw grounding boxes
    for box in boxes:
        y1, x1, y2, x2 = map(int, box)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)

    # annotate count
    cv2.putText(output,
                f"Count: {pre_cnt}",
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1.0,
                (0, 255, 255),
                2)

    return output, coord_list, pre_cnt