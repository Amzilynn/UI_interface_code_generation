from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
 
image_path = "C:/Users/User/Desktop/sketch-to-codeV0/YOLOV8_Approach/BackEnd/data/Manually annotated Dataset  V1 (under represented classes)/test/images/fr-fr_facebook_com_png_jpg.rf.c3304f4e40bca36e1e9ff230010a02f3.jpg"
img = Image.open(image_path)
 
# Resize image
img = img.resize((224, 224))  # Resize to target size (224, 224)D
 
# Convert image to numpy array and normalize
img_array = np.array(img) / 255.0
 
img_array = np.expand_dims(img_array, axis=0)
 
model = load_model("C:/Users/User/Downloads/modelclasssification2.h5")
 
prediction = model.predict(img_array)
predicted_class = "Sketch" if prediction[0][0] > 0.5 else "Screenshot"
 
print("Predicted class:", predicted_class)