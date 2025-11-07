
compile:
	nvcc -O3 -rdc=true src/util.cu src/bloomFilter.cu src/greedySearch.cu src/outNeighbors.cu src/reverseEdge.cu src/vamana.cu -o bin/vamana

compile-debug:
	nvcc -G -g -rdc=true src/util.cu src/bloomFilter.cu src/greedySearch.cu src/outNeighbors.cu src/reverseEdge.cu src/vamana.cu -o bin/vamana

build: 
	./bin/vamana data/sift10k_randomgraph.bin data/base.bin build/vamana.out

build-debug:
	cuda-gdb --args ./bin/vamana data/sift10k_randomgraph.bin data/base.bin build/vamana.out
	python3 scripts/bang-preprocess.py build/vamana.out sift10kfiles/sift10k_index | sort | uniq -c


search10k:
	python3 scripts/bang-preprocess.py build/vamana.out sift10kfiles/sift10k_index
	./bin/bang_search sift10kfiles/sift10k_index sift10kfiles/siftsmall_query.bin sift10kfiles/sift10k_groundtruth.bin 100 10 float l2

search20k:
	python3 scripts/bang-preprocess.py build/vamana.out sift10kfiles/sift10k_index
	./bin/bang_search sift20kfiles/sift20k_index sift20kfiles/siftsmall_query.bin sift20kfiles/sift20k_groundtruth.bin 100 10 float l2

search1m:
	python3 scripts/bang-preprocess.py build/vamana.out sift1mfiles/sift1m_index --N=1000000
	./bin/bang_search sift1mfiles/sift1m_index sift1mfiles/sift_query.bin sift1mfiles/sift_groundtruth.bin 100 10 float l2


.PHONY: compile build build-debug search
