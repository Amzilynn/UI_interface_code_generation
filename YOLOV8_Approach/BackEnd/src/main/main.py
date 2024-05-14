import logging
from pathlib import Path
from ultralytics import YOLO

def train_and_evaluate_yolo(train_data_path, test_data_path, epochs=10):
 
   # Set up logging
   logging.basicConfig(filename='training.log', level=logging.INFO)
   
   try:
      # Load a pretrained model
      model = YOLO('C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/src/yolo weights/yolov8n.pt')
      
      # Train the model
      print("Start training...")
      model.train(
         data=train_data_path,
         epochs=epochs,
         project='C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/runs/Manually annotated train v3 (only grayscale)'
         
      )
      print("YOLO model trained successfully!")

   except Exception as e:
      logging.error(f"Training failed: {str(e)}")

if __name__ == "__main__":
   train_data_path = Path('C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/data/Manually annotated dataset V3 (only grayscale)/data.yaml')
   test_data_path = Path('C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/data/Manually annotated dataset V3 (only grayscale)/test/test.yaml')
   train_and_evaluate_yolo(train_data_path, test_data_path)



   '''' # Evaluate the model
      print("Start evaluation...")
      evaluation_results = model.evaluate(test_data_path)
      print("Evaluation Results:", evaluation_results)
   '''