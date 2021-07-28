import os
import argparse
from typing import Optional
from PIL import Image
from dogifier.utils.resource import get_imagenet_class_map
from dogifier.model import Dogifier


def main(
    image_dir: str, *,
    backbone_name: Optional[str] = None,
    weight_file: Optional[str] = None,
    batch_size: Optional[int] = 1
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

    class_map = get_imagenet_class_map()

    for i in range(0, len(image_list), batch_size):
        image_files = image_list[i:i+batch_size]
        image_batch = [ Image.open(image_file) for image_file in image_files ]
        thing_classes = model.classify(image_batch)
        names = [ class_map[int(thing_class)][1] for thing_class in thing_classes]
        print(names)


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
