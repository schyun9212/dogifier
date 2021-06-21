from math import floor
import os
import random
import shutil

def split_imagefolder(root, image_folder="train", split_ratio=0.05):
    image_folder = os.path.join(root, image_folder)
    assert os.path.exists(image_folder)
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

def rollback_imagefolder(root, src_folder="val", dst_folder="train"):
    image_folder = os.path.join(root, src_folder)
    assert os.path.exists(image_folder)

    for label in os.listdir(image_folder):
        src_dir = os.path.join(image_folder, label)
        dst_dir = os.path.join(root, dst_folder, label)
        assert os.path.exists(dst_dir)

        images = os.listdir(src_dir)
        for image in images:
            src_path = os.path.join(src_dir, image)
            dst_path = os.path.join(dst_dir, image)
            shutil.move(src_path, dst_path)
    
    os.rmdir(image_folder)
