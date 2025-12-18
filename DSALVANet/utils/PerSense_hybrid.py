import cv2
import numpy as np
import torch
import math
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from sklearn.cluster import DBSCAN

def apply_scoremap(image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=np.float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)


def hybrid_IDM(src_img, boxes, model_output, test_idx):
    """
    Combines contour-based and peak+watershed instance detection.
    Returns:
      - output: visualization image
      - coord_list: list of torch.Tensor([x, y]) union of detected points
      - pre_cnt: predicted count = ceil(sum of density)
    """
    # 1) Compute pre_cnt and density map
    sum_pred = torch.sum(model_output)
    pre_cnt = int(math.ceil(sum_pred))
    h_orig, w_orig = src_img.shape[:2]

    density_pred = (
        model_output.squeeze(0)
                    .permute(1, 2, 0)
                    .cpu()
                    .detach()
                    .numpy()
    )
    density_pred = cv2.resize(density_pred, (w_orig, h_orig))
    density_pred = (density_pred - density_pred.min()) / (density_pred.max() - density_pred.min())
    gray = (density_pred * 255).astype(np.uint8)
    gray_density = gray.copy()

    coord_set = set()
    eps = 1e-15

    # --- Approach A: Contour + distance-transform splitting
    _, thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(thresh, kernel, iterations=1)

    contours, _ = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

    # --- Approach B: peak_local_max
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
