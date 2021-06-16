import os
from PIL import Image
import argparse
import torch
import torchvision.transforms as T
import json

from dogifier.config import get_cfg
from dogifier.model import Dogifier


def main(args) -> None:
    cfg = get_cfg()
    if args.config_file is not None:
        cfg.merge_from_file(args.config_file)

    model = Dogifier(cfg.MODEL)
    model.eval()

    image_dir = args.image_dir
    if os.path.isdir(image_dir):
        image_list = [ os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    else:
        image_list = [ image_dir ]

    transforms = T.Compose(
        [
            T.Resize((224, 224), 3),
            T.ToTensor(),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    with open("imagenet_class_index.json", 'r') as f:
        imagenet_class_idx = json.load(f)
        imagenet_idx_to_class = [ item[1] for item in imagenet_class_idx.values() ]

    for image_path in image_list:
        image = Image.open(image_path)
        image = transforms(image)
        image = image.unsqueeze(0)
        result = model(image)
        result = torch.argmax(result)
        print(imagenet_idx_to_class[int(result)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default=None)
    parser.add_argument("--image-dir", type=str, required=True)
    args = parser.parse_args()
    
    main(args)
