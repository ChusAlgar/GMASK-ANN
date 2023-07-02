from other_algorithms.FLANN.main_test_FLANN_config import FLANN


import sys
import re
import argparse
import logging
import ConfigParser
import io

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
    with open(configfile_path) as f:
        config_file = f.read()
    config = ConfigParser.RawConfigParser(allow_no_value=True)
    config.readfp(io.BytesIO(config_file))

    # According to the method choosen, carry out the experiment described on the configuration file

    if method == 'BruteForce':
        print("Please, use test_launcher_py3")

    elif method == 'KDTree':
        print("Please, use test_launcher_py3")

    elif method == 'MASK':
        print("Please, use test_launcher_py3")

    elif method == 'FLANN':
        FLANN(config)

    elif method == 'PYNN':
        print("Please, use test_launcher_py3")

    else:
        print("Method not able")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config_file", help="Config file | .ini", type=str)
    args = parser.parse_args()

    main(args)
