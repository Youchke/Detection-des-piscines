import cv2
import numpy as np
import os
import glob
from ultralytics import YOLO

# --- PARAMETERS (Adjust these for fine-tuning) ---
# Mozilla AI's pre-trained swimming pool detector (YOLO11m)
YOLO_MODEL_PATH = "model.pt"  # Downloaded from huggingface.co/mozilla-ai/swimming-pool-detector

# OpenCV refinement parameters (used within YOLO boxes)
LOWER_BLUE = np.array([85, 45, 30]) 
UPPER_BLUE = np.array([135, 255, 255]) 
MIN_AREA = 200 # Area within the crop
# -----------------------------------------------

def refine_contour_in_box(image, box_coords):
    """
    Given an image and a bounding box (x1, y1, x2, y2),
    returns the precise contour of the pool within that box using OpenCV.
    """
    x1, y1, x2, y2 = map(int, box_coords)
    # Add a small padding to the box if possible to avoid missing edges
    h, w = image.shape[:2]
    pad = 5
    x1_p, y1_p = max(0, x1 - pad), max(0, y1 - pad)
    x2_p, y2_p = min(w, x2 + pad), min(h, y2 + pad)

    # Crop the ROI
    roi = image[y1_p:y2_p, x1_p:x2_p]
    if roi.size == 0:
        return None

    # Preprocessing
    blurred = cv2.medianBlur(roi, 3)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
    
    # Color Thresholding
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    
    # Morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        # Fallback: if no blue zone found, return the box itself as a contour
        rect_cnt = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
        return rect_cnt

    # Take the largest contour that fits the "pool" criteria
    best_cnt = max(contours, key=cv2.contourArea)
    
    # Map contour back to global coordinates
    best_cnt[:, :, 0] += x1_p
    best_cnt[:, :, 1] += y1_p
    
    return best_cnt

def detect_pools(image_path, output_image_path, coordinates_path, model):
    """
    Detects swimming pools using YOLOv8 for localization and OpenCV for contour refinement.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image at {image_path}")
        return

    # 1. YOLOv8 Detection
    # Note: Using standard model might not find pools without custom training.
    # If the model is standard COCO, 'swimming pool' isn't a class.
    # However, we implement the logic for when the user provides a trained model.
    results = model(image, conf=0.25)
    
    pool_coords = []
    output_img = image.copy()
    pool_count = 0

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # If using custom model, check class: if box.cls[0] == POOL_CLASS_ID:
            
            # 2. Refine Contour
            coords = box.xyxy[0].cpu().numpy()
            refined_cnt = refine_contour_in_box(image, coords)
            
            if refined_cnt is not None:
                pool_count += 1
                pool_coords.append(refined_cnt.reshape(-1, 2).tolist())

                # Draw (Blue, width 1)
                cv2.drawContours(output_img, [refined_cnt], -1, (255, 0, 0), 1)
                
                # Label
                x, y, _, _ = map(int, coords)
                cv2.putText(output_img, f"Pool {pool_count}", (x, y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

    # Save outputs
    cv2.imwrite(output_image_path, output_img)
    print(f"Output saved to: {output_image_path}")

    # Append to coordinates file
    with open(coordinates_path, 'a') as f:
        for i, pool in enumerate(pool_coords):
            f.write(f"Pool {i+1}:\n")
            for pt in pool:
                f.write(f"{pt[0]},{pt[1]}; ")
            f.write("\n\n")
    print(f"Detected {len(pool_coords)} pools.")

if __name__ == "__main__":
    # Load YOLO Model
    model_path = os.path.join("..", "models", "model.pt")
    print(f"Loading YOLOv8 model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Define paths
    input_folder = os.path.join("..", "data", "input", "images", "swimming pool")
    output_folder = os.path.join("..", "data", "output")
    coordinates_file = os.path.join(output_folder, "coordinates.txt")

    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Search for all images
    image_files = glob.glob(os.path.join(input_folder, "*.*"))
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    image_files = [f for f in image_files if f.lower().endswith(valid_extensions)]

    if not image_files:
        print(f"No image found in {input_folder}.")
    else:
        if os.path.exists(coordinates_file):
            os.remove(coordinates_file)
            
        print(f"Found {len(image_files)} images. Starting batch processing...")
        
        for i, target_image in enumerate(image_files):
            base_name = os.path.basename(target_image)
            file_root = os.path.splitext(base_name)[0]
            current_output = os.path.join(output_folder, f"output_{file_root}.jpg")
            
            print(f"[{i+1}/{len(image_files)}] Processing: {target_image}")
            
            with open(coordinates_file, 'a') as f:
                f.write(f"--- Image: {base_name} ---\n")
                
            detect_pools(target_image, current_output, coordinates_file, model)
        
        print(f"\nBatch processing complete. Output images and coordinates.txt generated in {output_folder}.")

