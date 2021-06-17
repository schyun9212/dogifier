from math import floor
import os
import random
import shutil

def split_imagefolder(root, image_folder="train", split_ratio=0.2):
    image_folder = os.path.join(root, image_folder)
    val_dir = os.path.join(root, "val")
    os.mkdir(val_dir)
    
    for label in os.listdir(image_folder):
        src_dir = os.path.join(image_folder, label)
        dst_dir = os.path.join(val_dir, label)
        os.mkdir(dst_dir)

        images = os.listdir(src_dir)
        val_images = random.sample(images, floor(len(images) * split_ratio))

        for val_image in val_images:
            src_path = os.path.join(src_dir, val_image)
            dst_path = os.path.join(dst_dir, val_image)
            shutil.move(src_path, dst_path)
