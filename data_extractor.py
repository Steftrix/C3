import os
import requests
import tarfile
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET

dataset_path = "Date"
images_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
annotations_url = "http://vision.stanford.edu/aditya86/ImageNetDogs/annotation.tar"
images_dir = os.path.join(dataset_path, 'Images')
annotations_dir = os.path.join(dataset_path, 'Annotation')

def download_and_extract(url, path):
    if not os.path.exists(path):
        os.mkdir(path)
    packet_file = os.path.join(path, os.path.basename(url))
    if not os.path.exists(packet_file):
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(packet_file, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    if not os.path.exists(os.path.join(path, os.path.basename(url).split('.')[0])):
        with tarfile.open(packet_file) as tfile:
            tfile.extractall(path)
            
if not os.path.exists(images_dir):
    download_and_extract(images_url, dataset_path)
if not os.path.exists(annotations_dir):
    download_and_extract(annotations_url, dataset_path)

breed_list = os.listdir(annotations_dir)

fig = plt.figure(figsize=(15,8))
for i in range(15):
    axs = fig.add_subplot(3,5,i+1)
    breed = np.random.choice(breed_list) 
    dog = np.random.choice(os.listdir(os.path.join(annotations_dir, breed)))  
    img = Image.open(os.path.join(images_dir, breed, dog + '.jpg'))
    tree = ET.parse(os.path.join(annotations_dir, breed, dog)) 
    root = tree.getroot()
    object_1 = root.findall('object')[0]
    name = object_1.find('name').text
    axs.set_title(name)
    plt.imshow(img)
    plt.axis('off')

plt.suptitle("Exemplu imagini câini")
plt.show()
