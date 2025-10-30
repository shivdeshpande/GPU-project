import struct
import sys
import numpy
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Process a binary file and node index.")
parser.add_argument("binary_file", type=str, help="Path to the binary file")
parser.add_argument("node_index", type=int, help="Index of the node")
parser.add_argument("--N", type=int, default=10000, help="Number of nodes (default: 10000)")
parser.add_argument("--DIM", type=int, default=128, help="Dimension size (default: 128)")
parser.add_argument("--DEGREE", type=int, default=64, help="Degree (default: 64)")
parser.add_argument("--DATATYPESIZE", type=int, default=4, help="Data type size in bytes (default: 4)")

args = parser.parse_args()

file_to_read = args.binary_file
idx = args.node_index
N = args.N
DIM = args.DIM
DEGREE = args.DEGREE
DATATYPESIZE = args.DATATYPESIZE

with open(file_to_read, "rb") as f:
    print("Number of  Nodes to discover: ", N)    
    NodesRead = 0

    for i in range(idx): 
        # read feature vector    
        a = f.read(DATATYPESIZE * DIM)

        # read degree
        deg_bytes = f.read(4)

        d = struct.unpack('<I', deg_bytes)[0]

        if(d > DEGREE):
            print(f"node {i}, degree {d} exceeds {DEGREE}")
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
                print(f"invalid neighbor id {neighborId}")
                exit()

        NodesRead = NodesRead + 1
        

    a = f.read(DATATYPESIZE * DIM)
    deg_bytes = f.read(4)
    d = struct.unpack('<I', deg_bytes)[0]
    if(d > DEGREE):
        print(f"node {i}, degree {d} exceeds {DEGREE}")
        exit()

    # read neighbor IDs
    Nbytes = f.read(DEGREE * 4) 
    neighborIds = struct.unpack('<' + 'I' * DEGREE, Nbytes)
    print(sorted(neighborIds[:d]))
