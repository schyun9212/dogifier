

class WordTree():
    def __init__(self, tree_file: str, labels_file: str, names_file: str) -> None:
        self.tree = []
        self.metadata = {}

        with open(names_file, 'r') as f:
            names = f.read().strip().split("\n")
        
        with open(tree_file, 'r') as f:
            nodes = f.read().strip().split("\n")

        for i, (node, name) in enumerate(zip(nodes, names)):
            label, to = node.split(" ")
            self.tree.append((label, int(to)))
            self.metadata[label] = (i, name)

    def is_descendant(self, descendant_label: str, ancestor_name: str):
        index, name = self.metadata[descendant_label]
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
