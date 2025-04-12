import sys
import os


def setup_path():
    current_file = os.path.abspath(__file__)
    root = os.path.abspath(os.path.join(current_file, '..', '..'))
    if root not in sys.path:
        sys.path.append(root)


setup_path()