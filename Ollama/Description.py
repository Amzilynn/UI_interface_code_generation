import ollama 
import pandas as pd
from PIL import Image
import os
from io import BytesIO

# Load the DataFrame from a CSV file, or create a new one if the file doesn't exist
def load_or_create_dataframe(filename):
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
    else:
        df = pd.DataFrame(columns=['image_file', 'description'])
    return df

df = load_or_create_dataframe('C:/Users/User/Desktop/sketch-to-codeV0/Ollama/image_descriptions.csv')
print('file created')

def process_image(image_path):
    print(f"\nProcessing {image_path}\n")
    with Image.open(image_path) as img:
        with BytesIO() as buffer:
            img.save(buffer, format='JPEG')  # Save as JPG
            image_bytes = buffer.getvalue()

    full_response = ''

    # Check if model is available locally, if not pull it first
    print("Downloading model 'llava:13b-v1.6'...")
    ollama.pull('llava:13b-v1.6')
    print("Download completed")

    print("Starting the generation ...")
    # Generate a description of the image
    for response in ollama.generate(model='llava:13b-v1.6',
                                    prompt='''What's in this image''',
                                    images=[image_bytes],
                                    stream=False):
        # Print the response to the console and capture it
        print(response, end='', flush=True)
        full_response += response

    # Add the description to the DataFrame (assuming 'description' is a column)
    df = df.append({'image_file': image_path, 'description': full_response}, ignore_index=True)

# Provide the path to your image file
image_path = "C:/Users/User/Desktop/sketch-to-codeV0/Ollama/images/firebase.jpg"

# Process the image and update DataFrame
process_image(image_path)


