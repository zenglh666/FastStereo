import argparse
import os
import time
import sys
import subprocess
from datetime import datetime


parser = argparse.ArgumentParser(description='Queue')

parser.add_argument('--file', type=str, default="",
                    help='select file')
parser.add_argument('--sleep', type=int, default=10,
                    help='select file')

def main():
    args = parser.parse_args()
    if args.file == "":
        raise ValueError("no input file")
    
    past = [""]
    while True:
        time.sleep(args.sleep)
        with open(args.file, 'r') as f:
            now = f.readlines()
        if now != past:
            for l in now:
                print(l)
                retcode = subprocess.call(l, shell=True)
                print(retcode)
            past = now

if __name__ == '__main__':
   main()
    
