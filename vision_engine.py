import cv2
import numpy as np

def analyze_surface_rot(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return 0.0, None
    
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 1. Mask out pure white/bright backgrounds
    non_bg_mask = cv2.inRange(img_hsv, np.array([0, 20, 20]), np.array([180, 255, 255]))
    fruit_pixel_count = np.sum(non_bg_mask > 0)
    
    if fruit_pixel_count == 0:
        return 0.0, img

    # 2. Detect dark, discolored, or low-saturation defect areas
    lower_defect = np.array([0, 0, 0])
    upper_defect = np.array([180, 255, 120])
    defect_mask = cv2.inRange(img_hsv, lower_defect, upper_defect)
    
    # Combine masks to only count defects ON the fruit
    fruit_defects = cv2.bitwise_and(defect_mask, defect_mask, mask=non_bg_mask)
    
    # 3. Draw Red Bounding Boxes around detected defect contours for Explainable AI (XAI)
    annotated_img = img.copy()
    contours, _ = cv2.findContours(fruit_defects, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for c in contours:
        if cv2.contourArea(c) > 30: # Filter small noise
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(annotated_img, "DEFECT", (x, max(10, y - 5)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    defect_ratio = np.sum(fruit_defects > 0) / fruit_pixel_count
    
    # Convert BGR (OpenCV) back to RGB for Streamlit rendering
    annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    
    return round(float(defect_ratio), 2), annotated_img_rgb