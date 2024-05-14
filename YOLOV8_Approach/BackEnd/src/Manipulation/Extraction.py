import ultralytics
from ultralytics import YOLO
import cv2
import os

# Load YOLOv8 model
model = YOLO('C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/runs/Manuallyannotatedtrainv3onlygrayscale/train/weights/best.pt')

# Load input image (replace with your own image path)
input_image_path = "C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/data/Manually annotated dataset V3 (only grayscale)/test/images/000029_jpg.rf.584057ad8aec6f5b685ca6480e0f4773.jpg"
img = cv2.imread(input_image_path)


# Perform inference
results = model.predict(img , show = True, save=True, project="C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/results", exist_ok=True)


# Create a directory for saving results 
output_dir = "C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/Backend/results"
os.makedirs(output_dir, exist_ok = True)

element_counter = 0

# Extract bounding box information
for r in results:
  for box in r.boxes:
    # Get box coordinates (left, top, right, bottom)
    x1, y1, x2, y2 = box.xyxy[0]
    # Get class label
    class_label = model.names[int(box.cls)]
    # Assign a unique element ID 
    element_counter += 1  # Increment counter for each box
    element_id = f"element_{element_counter}"

    # Calculate width and height
    width = x2 - x1
    height = y2 - y1

    # Create CSS styles
    css_style = f"""
    .element_{element_id} {{
        position: absolute;
        top: {y1}px;
        left: {x1}px;
        width: {width}px;
        height: {height}px;
        border: 1px solid red;
    }}
    """


    # Create text file path
    output_filename = os.path.splitext(os.path.basename(input_image_path))[0] + ".txt"
    text_file_path = os.path.join(output_dir, output_filename)

    # Write bounding box information and CSS to formatted_data.txt
    with open(text_file_path, 'a') as f:
      f.write(f"Element_id : {element_id} Class :{class_label} Top-Left (x1, y1) :{x1} , {y1} bottom-right (x2, y2) :{x2} ,{y2}\n")
      f.write(f"Details\n")
      f.write(f"Object type: {box.cls}\n")
      f.write(f"Coordinates: {box.xyxy}\n")
      f.write(f"Probability: {box.conf}\n")
      f.write(f"CSS Styles:\n{css_style}\n")
      f.write("\n")

