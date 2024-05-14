from itertools import count
from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/runs/Manually annotated grayscale + resizing( 10epochs)/train/weights/best.pt')

# Mapping of class indices to class names
class_names = ['Button', 'Checkbox', 'Image', 'Input-Field', 'Radio-Button', 'Search-Box', 'Select-Dropdown', 'Text', 'icon']

# Counter for generating unique IDs
id_counter = count(start=1)

def process_image(image_path, formatted_data_file):

    results = model(image_path)  # Run inference
    print(results)

    # Get and format the bounding box data for each detected object
    formatted_data = []
    for result in results:
        for box in result.boxes.data:
            x1, y1, x2, y2, conf, cls = box.tolist()  # Extract coordinates, confidence, and class
            cls = int(cls)
            class_name = class_names[cls]
            unique_id = next(id_counter)  # Get the next unique ID
            formatted_data.append(f"The element number {unique_id} :\n  is a : {class_name} \n  Positions are : x1 {x1:.2f} y1 {y1:.2f} x2 {x2:.2f} y2 {y2:.2f}\n\n ")
            

    # Save the visualized result image as result.jpg
    result.save(filename='C:/Users/User/Desktop/sketch-to-codeV0/src/testing/test-results/result.jpg')

    # Write the formatted data to formatted_data.txt
    with open(formatted_data_file, 'w') as f:
        f.writelines(formatted_data)

# Process a new image
image_path = "C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/FrontEnd/uploads/image.jpg"
formatted_data_file = 'C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/FrontEnd/results/formatted_data.txt'
process_image(image_path, formatted_data_file)



#detect 
'''yolo task=detect mode=predict model=C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/runs/Manuallygrayscale+resizing/train/weights/best.pt  source="C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/FrontEnd/uploads/000005_png_jpg.rf.70d579b9b7a4508ea5e88a3812e06b4b.jpg'''