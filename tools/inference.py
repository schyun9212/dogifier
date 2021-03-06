import os
import argparse
from typing import Optional
from PIL import Image
import numpy as np

from dogifier.model import Dogifier


def main(
    image_dir: str, *,
    backbone_name: Optional[str] = None,
    weight_file: Optional[str] = None,
    batch_size: Optional[int] = 1,
    wordtree_target: Optional[str] = None,
    to_name: Optional[bool] = False,
    device: Optional[str] = "cpu",
    output_dir: Optional[str] = "output/inference"
) -> None:
    assert backbone_name or weight_file, "backbone name or weight file should be specified"

    if weight_file:
        model = Dogifier.load_from_checkpoint(weight_file)
    else:
        model = Dogifier(backbone_name, num_classes=1000, freeze=True)
    model.eval()
    model.to(device)

    if os.path.isdir(image_dir):
        image_list = [ os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    else:
        image_list = [ image_dir ]

    results = []
    for i in range(0, len(image_list), batch_size):
        image_files = image_list[i:i+batch_size]
        image_batch = [ Image.open(image_file).convert("RGB") for image_file in image_files ]
        results += model.classify(image_batch, to_name=to_name, wordtree_target=wordtree_target)

    print(results)

    if wordtree_target:
        import shutil
        os.makedirs(output_dir, exist_ok=True)

        target_dir = os.path.join(output_dir, backbone_name,  "target")
        non_target_dir = os.path.join(output_dir, backbone_name, "non-target")
        os.makedirs(target_dir, exist_ok=True)
        os.makedirs(non_target_dir, exist_ok=True)

        for image_file, result in zip(image_list, results):
            image_name = os.path.basename(image_file)
            if result:
                dst_path = os.path.join(target_dir, image_name)
            else:
                dst_path = os.path.join(non_target_dir, image_name)  
            shutil.copyfile(image_file, dst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-dir", type=str, required=True)
    parser.add_argument("--backbone-name", type=str, default=None)
    parser.add_argument("--weight-file", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--wordtree-target", type=str, default=None)
    parser.add_argument("--to-name", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    main(
        args.image_dir,
        backbone_name=args.backbone_name,
        weight_file=args.weight_file,
        batch_size=args.batch_size,
        wordtree_target=args.wordtree_target,
        to_name=args.to_name,
        device=args.device
    )
