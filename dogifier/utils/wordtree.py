from typing import Union

from .resource import (
    get_wordtree,
    get_wordtree_names,
    get_imagenet_class_map
)


class WordTree():
    def __init__(self) -> None:
        self.tree = get_wordtree()
        self.imagenet_class_map = get_imagenet_class_map()
        self.metadata = {}

        names = get_wordtree_names()

        for i, (node, name) in enumerate(zip(self.tree, names)):
            label, _ = node
            self.metadata[label] = (i, name)

    def search_ancestor(
        self,
        descendant: Union[str, int],
        ancestor_name: str
    ) -> bool:
        if isinstance(descendant, int):
            descendant = self.to_label(descendant)

        index, name = self.metadata[descendant]
        label, to = self.tree[index]

        safe_cnt = 0
        while(
            to > -1 and \
            name != ancestor_name \
            and safe_cnt < len(self.tree)
        ):
            label, to = self.tree[to]
            index, name = self.metadata[label]

            safe_cnt += 1
        
        return name == ancestor_name

    def to_name(self, index: int):
        _, name = self.imagenet_class_map[index]
        return name

    def to_label(self, index: int):
        label, _ = self.imagenet_class_map[index]
        return label
