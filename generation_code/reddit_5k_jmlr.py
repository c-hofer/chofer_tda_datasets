from generation.reddit_graph import run
from generation.path_config import data_raw_path, data_generated_path
import os.path as pth


if __name__ == '__main__':
    raw_data_path = pth.join(data_raw_path, 'reddit_multi_5K.graph')
    output_path = pth.join(data_generated_path, 'reddit_5k_jmlr.h5')
    run(raw_data_path, output_path, 10)