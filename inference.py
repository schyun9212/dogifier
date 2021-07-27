import os
from PIL import Image
import argparse
import torch
import torchvision.transforms as T
import json
from typing import Optional

from dogifier.model import Dogifier


def build_imagenet_class_map():
    with open("data/imagenet_class_index.json", 'r') as f:
        imagenet_class_idx = json.load(f)
        imagenet_idx_to_class = [ item[1] for item in imagenet_class_idx.values() ]
    return imagenet_idx_to_class


def build_transform():
    transforms = T.Compose(
        [
            T.Resize((224, 224), 3),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )
    return transforms


def main(
    image_dir: str, *,
    backbone_name: Optional[str] = None,
    weight_file: Optional[str] = None
) -> None:
    assert backbone_name or weight_file, "backbone name or weight file should be specified"

    if weight_file:
        model = Dogifier.load_from_checkpoint(weight_file)
    else:
        model = Dogifier(backbone_name, num_classes=1000, freeze=True)
    model.eval()

    image_dir = args.image_dir
    if os.path.isdir(image_dir):
        image_list = [ os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    else:
        image_list = [ image_dir ]

    transforms = build_transform()
    class_map = build_imagenet_class_map()

    for image_path in image_list:
        image = Image.open(image_path)
        image = transforms(image)
        image = image.unsqueeze(0)
        result = model(image)
        result = torch.argmax(result)
        print(class_map[int(result)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--weight-file", type=str, default=None)
    args = parser.parse_args()
    
    main(
        args.image_dir,
        backbone_name=args.backbone_name,
        weight_file=args.weight_file
    )
