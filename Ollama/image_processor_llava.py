"""from ollama import generate
import pandas as pd
import ollama
from PIL import Image
from io import BytesIO
import os

# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(filename):
  if os.path.isfile(filename):
    df = pd.read_csv(filename)
  else:
    df = pd.DataFrame(columns=['image_file', 'description'])
  return df

# Function to process image and save description to text file
def process_image(image_path, text_file_path):
  # Ensure the directories leading to the text file exist
  os.makedirs(os.path.dirname(text_file_path), exist_ok=True)

  with Image.open(image_path) as img:
    with BytesIO() as buffer:
      img.save(buffer, format='JPEG') 
      image_bytes = buffer.getvalue()

  full_response = ''

  # Check if model is available locally, if not pull it first
  ollama.pull('llava:13b-v1.6')

  # Generate a description of the image
  for response in generate(model='llava:13b-v1.6',
               prompt='''what's in this image , decribe it with details''' + image_path ,
               images=[image_bytes],
               stream=False):
    full_response += response

  # Write the description to a text file
  with open(text_file_path, 'w') as file:
    file.write(full_response)

# Provide the path to your image file and text file
image_path = "C:/Users/User/Desktop/sketch-to-codeV0/Ollama/images/girl eating.jpg"
text_file_path = "C:/Users/User/Desktop/sketch-to-codeV0/Ollama/descriptions/girl_eating_description.txt"

# Process the image and save description to text file
process_image(image_path, text_file_path)

print("Description saved to:", text_file_path)"""
from transformers import pipeline
from PIL import Image


model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id)


def process_image(image_path):
    try:
        image = Image.open(image_path)
        prompt = "Describe this image:"
        description = pipe(prompt=prompt, images=image_path)[0]["generated_text"]

        return description
    except Exception as e:
        return f"Error processing {image_path}: {str(e)}"

# Example usage:
image_path = "C:/Users/User/Desktop/sketch-to-codeV0/Ollama/images/girl eating.jpg"
description = process_image(image_path)

# Save the description to a text file
with open("image_description.txt", "w") as file:
    file.write(description)
