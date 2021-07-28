import os
import json


resource_dir = os.path.join(os.path.dirname(__file__), "../resource")

IMAGENET_CLASS_INDEX_FILE = os.path.join(resource_dir, "imagenet_class_index.json")

WORDTREE_TREE_FILE = os.path.join(resource_dir, "9k.tree")
WORDTREE_NAMES_FILE = os.path.join(resource_dir, "9k.names")
WORDTREE_LABELS_FILE = os.path.join(resource_dir, "9k.labels")


def get_imagenet_class_map():
    with open(IMAGENET_CLASS_INDEX_FILE, 'r') as f:
        imagenet_class_idx = json.load(f)
        imagenet_idx_to_class = [ item for item in imagenet_class_idx.values() ]
    return imagenet_idx_to_class


def get_wordtree():
    with open(WORDTREE_TREE_FILE, 'r') as f:
        nodes = f.read().strip().split("\n")

    def process(node: str):
        label, to = node.split(" ")
        return (label, int(to))

    tree = [ process(node) for node in nodes ]
    return tree


def get_wordtree_names():
    with open(WORDTREE_NAMES_FILE, 'r') as f:
        names = f.read().strip().split("\n")
    return names


def get_wordtree_labels():
    with open(WORDTREE_LABELS_FILE, 'r') as f:
        labels = f.read().strip().split("\n")
    return labels
