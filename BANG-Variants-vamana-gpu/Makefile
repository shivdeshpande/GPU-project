
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


# Concurrent executor (parallel INSERT/DELETE/QUERY)
compile-concurrent:
	nvcc -O3 -rdc=true -std=c++17 -Xcompiler -fopenmp \
		src/dynamic/concurrent_main.cu \
		src/dynamic/concurrent_executor.cu \
		src/dynamic/background_consolidate.cu \
		src/dynamic/workload.cpp \
		src/dynamic/insert.cu \
		src/dynamic/consolidate.cu \
		src/dynamic/deleteList.cu \
		src/util.cu \
		src/bloomFilter.cu \
		src/greedySearch.cu \
		src/outNeighbors.cu \
		src/reverseEdge.cu \
		-lpthread -lgomp \
		-o bin/concurrent_fresh_diskann

compile-concurrent-debug:
	nvcc -G -g -rdc=true -std=c++17 -Xcompiler -fopenmp \
		src/dynamic/concurrent_main.cu \
		src/dynamic/concurrent_executor.cu \
		src/dynamic/background_consolidate.cu \
		src/dynamic/workload.cpp \
		src/dynamic/insert.cu \
		src/dynamic/consolidate.cu \
		src/dynamic/deleteList.cu \
		src/util.cu \
		src/bloomFilter.cu \
		src/greedySearch.cu \
		src/outNeighbors.cu \
		src/reverseEdge.cu \
		-lpthread -lgomp \
		-o bin/concurrent_fresh_diskann

# Run concurrent executor with test workload
run-concurrent:
	./bin/concurrent_fresh_diskann build/vamana_alpha1.2.out test/mixed_50_50_100.jsonl \
		data/siftsmall_query.bin data/siftsmall_groundtruth.bin \
		--searchL 100 --workers 8 --batch 50

.PHONY: compile build build-debug search compile-concurrent compile-concurrent-debug run-concurrent
