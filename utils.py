import os

def make_directory_if_no_exists(directory):
    if not os.path.isdir(directory):
        os.mkdir(directory)