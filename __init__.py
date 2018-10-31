import os
from settings import edge_folder

if not os.path.isdir(edge_folder):
    os.mkdir(edge_folder)
