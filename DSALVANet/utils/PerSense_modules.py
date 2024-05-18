import cv2
import numpy as np
import torch
import matplotlib.pyplot  as plt
import math
import os

def apply_scoremap(image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=np.float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def IDM(src_img, boxes, model_output, test_idx):
        sum = torch.sum(model_output)
        pre_cnt = int(math.ceil(sum))
        h_orig, w_orig = src_img.shape[0], src_img.shape[1]
        density_pred = model_output.squeeze(0)
        density_pred = density_pred.permute(1, 2, 0)  # c x h x w -> h x w x c
        density_pred = density_pred.cpu().detach().numpy()
        density_pred = cv2.resize(density_pred, (w_orig, h_orig))
        density_pred = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
        density_pred= np.stack((density_pred,) * 3, axis=-1)


        # density_pred = cv2.cvtColor(density_pred, cv2.COLOR_RGB2BGR)
        # density_map_path = "./outputs/" + "density_map"
        # if not os.path.exists(density_map_path):
        #         os.mkdir('./outputs/density_map')
        # density_map_path = os.path.join(density_map_path, f'{test_idx}.png')
        # cv2.imwrite(density_map_path,255*density_pred)

        ##INSTANCE DETECTION MODULE

        # read input image
        # img = cv2.imread(density_map_path)

        # convert to grayscale
        # Convert the density map to grayscale directly
        gray = cv2.cvtColor((density_pred * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # threshold to binary
        thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

        kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) 
        eroded = cv2.erode(thresh, kernel1, iterations=1)
        contour_img = density_pred.copy()
        contours = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        contour_areas = [cv2.contourArea(contour) for contour in contours]

        # Calculate mean and standard deviation of contour areas
        mean_area = np.mean(contour_areas)
        std_area = np.std(contour_areas)

        # Set threshold as a multiple of standard deviation above the mean area
        threshold_multiplier = 2.0  # Adjust multiplier as needed
        adaptive_threshold = mean_area + threshold_multiplier * std_area

        # Identify contours whose areas exceed the adaptive threshold
        composite_contours = [contour for contour, area in zip(contours, contour_areas) if area > adaptive_threshold]
        cluster_count = 0
        coord_list = []
        eps = 1e-15

        for large_contour in composite_contours:
                # Create a mask for the large contour region
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [large_contour], -1, (255, 255, 255), thickness=cv2.FILLED)
                
                # Apply distance transform to the contour region
                dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
                
                # Threshold the distance transform to obtain markers for watershed segmentation
                _, thresholded_dist = cv2.threshold(dist_transform, 0.5*dist_transform.max(), 255, cv2.THRESH_BINARY)

                thresholded_dist = np.uint8(thresholded_dist)
                
                # Find contours within the segmented objects
                child_contours,_ = cv2.findContours(thresholded_dist, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # child_contours = child_contours[0] if len(child_contours) == 2 else child_contours[1]
                
                # Calculate and print the center of each child contour
                for child_contour in child_contours:
                        # Calculate the moments of the contour
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
                # cv2.putText(contour_img, "center", (cX - 20, cY - 20),
		# cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                cluster_count = cluster_count + 1
                coord_list.append(coords)
                       
        
        # print('Count of Objects',pre_cnt)

        # # save result
        # blob_img_path = "./outputs/" + "contours_img"
        # if not os.path.exists(blob_img_path):
        #         os.mkdir('./outputs/contours_img')
        # blob_img_path = os.path.join(blob_img_path, f'{test_idx}.png')
        # cv2.imwrite(blob_img_path, 255*contour_img)



        output = apply_scoremap(src_img, density_pred)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        for box in boxes:
            y1, x1, y2, x2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(output, "Result:{0}".format(pre_cnt), (10,30), cv2.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 255), 2)
        return output , coord_list, pre_cnt