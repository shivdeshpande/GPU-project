all: build run
   
build:
	nvcc -I"${CONDA_PREFIX}"/include -L"${CONDA_PREFIX}"/lib -DLIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE -lrmm -lcuvs main-1m.cu

run:
	LD_LIBRARY_PATH="${CONDA_PREFIX}/lib" ./a.out
