from PIL import Image
import os
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

checkdir = '../data/imagenet/train'
files = os.listdir(checkdir)
format = [".JPG", ".JPEG", '.jpg', 'jpeg']

for(path, dirs, f) in os.walk(checkdir):
    for file in f:
        if file.endswith(tuple(format)):
            try:
                class_path, _ = file.split('_')
                image = Image.open(class_path+"/"+file).load()
            except Exception as e:
                print("An exception is raised:", e)
                print(file)