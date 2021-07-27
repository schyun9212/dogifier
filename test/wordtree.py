import os

from dogifier.wordtree import WordTree

DATA_DIR = "/home/appuser/dogifier/data"
TREE_FILE = os.path.join(DATA_DIR, "9k.tree")
LABELS_FILE = os.path.join(DATA_DIR, "9k.labels")
NAMES_FILE = os.path.join(DATA_DIR, "9k.names")

wordtree = WordTree(TREE_FILE, LABELS_FILE, NAMES_FILE)

ancestor = "dog"
descendant = "n02112018" # Pomeranian
check = wordtree.is_descendant(descendant, ancestor)
print ("Done")