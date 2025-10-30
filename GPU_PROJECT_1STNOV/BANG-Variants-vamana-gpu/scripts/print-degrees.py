import struct
import sys
import numpy
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process Vamana binary files and index metadata.")
parser.add_argument("bin", type=str, help="Input binary file")
parser.add_argument("--N", type=int, default=10000, help="Number of nodes (default: 10000)")
parser.add_argument("--DIM", type=int, default=128, help="Dimension size (default: 128)")
parser.add_argument("--DEGREE", type=int, default=64, help="Degree R (default: 64)")
parser.add_argument("--DATATYPESIZE", type=int, default=4, help="Data type size in bytes (default: 4)")

args = parser.parse_args()

bin_file = args.bin
N = args.N
DIM = args.DIM
DEGREE = args.DEGREE
DATATYPESIZE = args.DATATYPESIZE

bin = open(bin_file, 'rb')

for i in range(N):
    bin.read(DIM * DATATYPESIZE)
    # bin.read(int(DATATYPESIZE))
    a=bin.read(4)
    d=struct.unpack('<I',a)[0]
    print(d)
    bin.read(4*DEGREE)

bin.close()
