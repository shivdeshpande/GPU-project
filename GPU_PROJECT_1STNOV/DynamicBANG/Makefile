# DynamicBANG Makefile
# GPU-Accelerated FreshDiskANN Implementation

# Compiler
NVCC = nvcc

# Flags
NVCC_FLAGS = -O3 -std=c++14 -Xcompiler -fopenmp
ARCH = -arch=sm_80  # Adjust for your GPU (sm_70, sm_75, sm_80, sm_86, sm_89)

# Dataset selection (default: SIFT10K)
DATASET ?= SIFT10K_DATASET

# Include paths
INCLUDES = -I. -I./utils

# Source files
SOURCES = dynamicBANG.cu delete.cu insert.cu consolidate.cu workload_simple.cu metrics.cu main.cu

# Object files
OBJECTS = $(SOURCES:.cu=.o)

# Target executable
TARGET = dynamicBANG

# Default target
all: $(TARGET)

# Link
$(TARGET): $(OBJECTS)
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -o $@ $^ -lcublas

# Compile
%.o: %.cu
	$(NVCC) $(NVCC_FLAGS) $(ARCH) -D$(DATASET) $(INCLUDES) -c $< -o $@

# Clean
clean:
	rm -f $(OBJECTS) $(TARGET) *.o

# Clean all
cleanall: clean
	rm -f dynamicBANG_metrics.txt

# Help
help:
	@echo "DynamicBANG Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all       - Build DynamicBANG (default)"
	@echo "  clean     - Remove object files and executable"
	@echo "  cleanall  - Remove all generated files"
	@echo "  help      - Show this help message"
	@echo ""
	@echo "Dataset Options (set DATASET=):"
	@echo "  SIFT10K_DATASET   - SIFT 10K vectors (default)"
	@echo "  SIFT1M_DATASET    - SIFT 1M vectors"
	@echo "  SIFT100M_DATASET  - SIFT 100M vectors"
	@echo ""
	@echo "Examples:"
	@echo "  make                          # Build with SIFT10K"
	@echo "  make DATASET=SIFT1M_DATASET   # Build with SIFT1M"
	@echo "  make ARCH=-arch=sm_75         # Build for specific GPU architecture"

.PHONY: all clean cleanall help
