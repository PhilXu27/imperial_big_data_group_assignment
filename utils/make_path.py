from os.path import exists
from os import mkdir
from utils.path_info import *


def _check_dirs(path):
    for _ in path:
        if not exists(_):
            mkdir(_)
    return


def make_path():
    check_list = [raw_data_path, demo_data_path]
    _check_dirs(check_list)
    return


if __name__ == '__main__':
    make_path()
