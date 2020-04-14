import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Instantiates a client
client = vision.ImageAnnotatorClient()

# The name of the image file to annotate
file_name = os.path.abspath('new_fruit_dataset/pi_data/063_Other_830.jpeg')

# Loads the image into memory
with io.open(file_name, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)

# Performs label detection on the image file
response = client.label_detection(image=image)
labels = response.label_annotations
print('Labels:')
for label in labels:
    print(label.description)

print('Text')
texts = response.text_annotations
print(len(texts))
for text in texts:
    print(text)