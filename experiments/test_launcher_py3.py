from other_algorithms.BruteForce.main_test_bruteforce_config import BruteForce
from other_algorithms.KDTree.main_test_KDTree_config import KDTree
from mask.experiments.test_knn.main_test_knn_radius_config import MASK
from other_algorithms.Pynndescent.main_test_PYNN_config import PYNN


import re
import argparse
import configparser

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


    # Get the path of the configuration file provided by the user
    config_file = args.config_file
    dataset = re.split('_|\.', config_file)[2]
    method = re.split('_|\.',  config_file)[5]
    configfile_path = "./experiments/config/" + dataset + "/" + config_file
    print("--- Reading " + config_file + " ---")


    # Open the configuration file
    config = configparser.ConfigParser()
    config.read(configfile_path)

    # According to the method choosen, carry out the experiment described on the configuration file

    if method == 'BruteForce':
        BruteForce(config)

    elif method == 'KDTree':
        KDTree(config)

    elif method == 'MASK':
        MASK(config)

    elif method == 'FLANN':
        print("Please, use test_launcher_py2")

    elif method == 'PYNN':
        PYNN(config)

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
