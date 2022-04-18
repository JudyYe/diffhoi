from sys import argv
import yaml
import os
import os.path as osp


def correct_yaml(filename):
    # '/glusterfs/yufeiy2/vhoi/output/100doh/camdelta_oThlearn/config.yaml'
    with open(filename) as fp:
        for line in fp:
            print(line)
    return 
    
if __name__ == '__main__':
    filename = argv[1]
    correct_yaml(filename)