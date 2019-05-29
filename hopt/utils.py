import os


def create_dirs(paths_list):
    for e in paths_list:
        os.makedirs(e, exist_ok=True)
