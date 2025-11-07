# This script converts file format from .vecs (fvecs, ivecs bvecs etc) to .bin format.
# Some datasets (e.g. SIFT http://corpus-texmex.irisa.fr/) are shared in vecs format. BANG needs data in bin for,at.

import struct
import sys

j=0;

if len(sys.argv) != 6:
    print("Check script command line params : <input vecs fils> <output bin file> <datatype in bytes> <dataset size> <dimenstions>")
    print("Example: To convert SIFT10K dataset: python vecs_to_binary.py siftsmall_base.fvecs base.bin 4 10000 128")
    exit()

file_to_read = sys.argv[1]  # VECS
file_to_write = sys.argv[2]  # BIN

DATATYPESIZE = int(sys.argv[3])    # Datatype size, 1 for bvecs, 4 for fvecs or ivecs
NUM = int(sys.argv[4])    # No of points
DIM = int(sys.argv[5])    # DIM

w= open(file_to_write, "wb")            #Output file
w.write(struct.pack('<I',NUM))
w.write(struct.pack('<I',DIM))

with open(file_to_read, "rb") as f:   #Input file
    while(j<NUM):
        a=f.read(4)
        d =struct.unpack('<I',a)[0]

        if(d!=DIM):
            print("crap")
            exit()
        for i in range(d):
            a=f.read(DATATYPESIZE)
            w.write(a)

        j=j+1
        if (j % 10000 == 0):
            print(j)

f.close()
w.close()
