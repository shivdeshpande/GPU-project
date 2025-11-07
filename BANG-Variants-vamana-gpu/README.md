# Vamana on GPU

## Instructions for Compiling

The code can be compiled using the command
```
make compile
```

There are several parameters for graph construction, which can be modified by editing the `vamana.h` file.

The program takes three arguments as input
```
./vamana <randomgraph> <basepoints> <output>
```

The program reads `NUM_QUERIES` points and their neighbors from the randomgraph, then it reads `N - NUM_QUERIES`
points from the basepoints file, which don't have any neighbors. The graph build algorithm is done on this graph.


The output graph requires some preprocessing to be compatible with BANG search. `scripts/bang_preprocess.py` is used
for this purpose. It is run as
```
python bang-preprocess.py [-h] [--N N] [--DIM DIM] [--DEGREE DEGREE] [--MEDOID MEDOID] [--DATATYPE DATATYPE] [--DATATYPESIZE DATATYPESIZE]
                          bin prefix
```

where `bin` is the output from Vamana and `prefix` is the BANG prefix for the dataset files. The Python script writes the
index and metadata files with the prefix.

## Exact Commands for SIFT10K

First, download the files for the SIFT10k dataset and  place them in a directory called `sift10kfles`.

The SIFT10K dataset files are in vecs, BANG needs files in bin format. Convert from vecs to bin using the below exaple
```
python3 scripts/vecs_to_binary.py siftsmall_base.fvecs data/base.bin 4 10000 128
python3 scripts/vecs_to_binary.py siftsmall_query.fvecs siftsmall_query.bin 4 100 128
python3 scripts/vecs_to_binary.py siftsmall_groundtruth.ivecs siftsmall_groundtruth.bin 4 100 100
```


Download the SIFT10K randomgraph file and place it in `data/sift10k_randomgraph.bin`, and the SIFT basepoints file
in `data/base.bin`.

To compile Vamana, run the command
```
nvcc -rdc=true src/util.cu src/bloomFilter.cu src/greedySearch.cu src/outNeighbors.cu src/reverseEdge.cu src/vamana.cu -o bin/vamana
```

This creates the executable `./bin/vamana`.

To run Vamana, use
```
./bin/vamana data/sift10k_randomgraph.bin data/base.bin build/vamana.out`
```

The resulting graph is written to `build/vamana.out`.

To convert the graph to DiskANN format, use

```
python3 scripts/bang-preprocess.py build/vamana.out sift10kfiles/sift10k_index
```

This updates the index file `sift10kfiles/sift10k_index_disk.bin` and the metadata file `sift10kfiles/sift10k_index_metadata.bin`

To run BANG Search:

```
./bin/bang_search sift10kfiles/sift10k_index sift10kfiles/siftsmall_query.bin sift10kfiles/sift10k_groundtruth.bin 100 10 float l2
```
Note: The query and groundtruth files in bin format are also available here: https://github.com/karthik86248/BANG-Billion-Scale-ANN/blob/main/sift10kfiles.tar.gz
BANG Search usage is documented here: https://github.com/karthik86248/BANG-Billion-Scale-ANN/blob/main/BANG_Base/ReadMe.pdf
