from other_algorithms.BruteForce.main_test_bruteforce_config import BruteForce
from other_algorithms.KDTree.main_test_KDTree_config import KDTree
from mask.experiments.test_knn.main_test_knn_radius_config import MASK
from other_algorithms.Pynndescent.main_test_PYNN_config import PYNN


import re
import argparse

'''
# Load the configuration file
config = configparser.ConfigParser()
config.read("./config/experiments.ini")

# Read experiments to carry out
dataset = config.get('experiments', 'dataset')
methods = config.get('experiments', 'methods')
distances = config.get('experiments', 'distances')
ks = config.get('experiments', 'ks')

ncentroids = config.getint('method', 'ncentroids')  # At MASK, ncentroids = tam_grupo*n_centroides = 8*16 = 128
algorithm = config.get('method', 'algorithm')  # Possible values 'auto', 'ball_tree', 'kd_tree', 'brute'
'''


def main(args):

    config_file = args.config_file

    method = re.split('_|\.',  config_file)[5]

    print("--- Reading " + config_file + " ---")


    if method == 'BruteForce':
        BruteForce(config_file)

    elif method == 'KDTree':
        KDTree(config_file)

    elif method == 'MASK':
        MASK(config_file)

    elif method == 'FLANN':
        print("Please, use test_launcher_py2")

    elif method == 'PYNN':
        PYNN(config_file)

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
