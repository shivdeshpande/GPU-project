import struct
import sys
import numpy
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process a binary file and number of nodes.")
parser.add_argument("binary_file", type=str, help="Path to the binary file")
parser.add_argument("--N", type=int, default=10000, help="Number of nodes (default: 10000)")
parser.add_argument("--DEGREE", type=int, default=64, help="Degree (default: 64)")
parser.add_argument("--DIM", type=int, default=128, help="Dimension size (default: 128)")
parser.add_argument("--DATATYPESIZE", type=int, default=4, help="Data type size in bytes (default: 4)")

args = parser.parse_args()

file_to_read = args.binary_file
N = args.N
DEGREE = args.DEGREE
DIM = args.DIM
DATATYPESIZE = args.DATATYPESIZE

with open(file_to_read, "rb") as f:
    print("Number of nodes to discover: ", N)    
    NodesRead = 0

    for i in range(N): 
        a = f.read(DATATYPESIZE * DIM) # read feature vector    
        deg_bytes = f.read(4) # read degree
        d = struct.unpack('<I', deg_bytes)[0]

        if(d > DEGREE):
            print(f"node {i}: degree {d} exceeds {DEGREE}")
            exit()
        
        # read neighbor IDs
        Nbytes = f.read(DEGREE * 4) 
        
        neighborIds = struct.unpack('<' + 'I' * DEGREE, Nbytes)

        # for i in range(d):
        i = 0
        for neighborId in neighborIds:
            i += 1
            if(i == d):
                continue

            if neighborId >= N or neighborId < 0:
                print(f"node {i}: invalid neighbor id {neighborId}")
                exit()

        NodesRead = NodesRead + 1
        
        # if (NodesRead % 10000 == 0):
        #     print(NodesRead)

print("Total # of Nodes Discovered = ", NodesRead)
