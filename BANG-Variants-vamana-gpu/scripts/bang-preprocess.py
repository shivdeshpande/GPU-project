import struct
import sys
import numpy
import numpy as np

import argparse

parser = argparse.ArgumentParser(description="Process Vamana binary files and index metadata.")
parser.add_argument("bin", type=str, help="Input binary file")
parser.add_argument("prefix", type=str, help="Output prefix")
parser.add_argument("--N", type=int, default=10000, help="Number of nodes (default: 10000)")
parser.add_argument("--DIM", type=int, default=128, help="Dimension size (default: 128)")
parser.add_argument("--DEGREE", type=int, default=64, help="Degree R (default: 64)")
parser.add_argument("--MEDOID", type=int, default=5000, help="Medoid flag (default: 5000)")
parser.add_argument("--DATATYPE", type=int, default=2, help="Data type code (default: 2 for float)")
parser.add_argument("--DATATYPESIZE", type=int, default=4, help="Data type size in bytes (default: 4)")

args = parser.parse_args()

bin_file = args.bin
index_file = args.prefix + '_disk.bin'
metadata_file = args.prefix + '_disk_metadata.bin'
N = args.N
DIM = args.DIM
DEGREE = args.DEGREE
MEDOID = args.MEDOID
DATATYPE = args.DATATYPE
DATATYPESIZE = args.DATATYPESIZE

# MEDOID = 126689

bin = open(bin_file, 'rb')
index = open(index_file, 'wb')
metadata = open(metadata_file, 'wb')


metadata.write(struct.pack('<Q', int(MEDOID)))
metadata.write(struct.pack('<Q', int(DIM * DATATYPESIZE + (DEGREE+1) * 4))) # Max node size
# metadata.write(struct.pack('<Q', int(64))) # Nodes per sec (unused)
metadata.write(struct.pack('<I', int(DATATYPE)))
metadata.write(struct.pack('<I', int(DIM)))
metadata.write(struct.pack('<I', int(DEGREE)))
metadata.write(struct.pack('<I', int(N)))

for i in range(N):
    for dim in range(DIM):
         b=bin.read(int(DATATYPESIZE))
         index.write(b) 
    a=bin.read(4)
    index.write(a)   
    d=struct.unpack('<I',a)[0]
    # print(d)
    if(d>DEGREE or d == 0) :
        print("crap")
        print(d)
        exit()
    arr = numpy.zeros(d,  dtype='<u4')
    for k in range(d):
        a=bin.read(4)
        #w.write(a)
        neighbour = struct.unpack("<I",a)[0] 
        #print("neighbour=", str(neighbour))
        arr[k] = neighbour                  
    #print(arr)
    arr_sorted = np.sort(arr)
    for l in range(d):
       index.write(struct.pack("<I",arr_sorted[l]))
    #print(arr_sorted)
    for kk in range(k+1,DEGREE):
        #print(kk)
        a=bin.read(4)
        index.write(a)
    
    if(i % 10000 == 0):
       print(i)

bin.close()
index.close()
metadata.close()
