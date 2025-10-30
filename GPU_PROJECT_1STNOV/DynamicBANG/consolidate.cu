#include "dynamicBANG.h"
#include "utils/utils.h"
#include "utils/timer.h"
#include <cstring>
#include <cstdio>

// ============================================================================
// CONSOLIDATION LOGIC
// ============================================================================

void initStaticIndex(StaticIndex* index, const char* index_file) {
    printf("[StaticIndex] Loading from %s...\n", index_file);

    // Get file size
    FILE* fp = fopen(index_file, "rb");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open index file %s\n", index_file);
        exit(1);
    }

    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    index->total_size_bytes = file_size;
    index->num_nodes = file_size / INDEX_ENTRY_LEN;
    index->capacity = N;  // From dataset configuration

    printf("[StaticIndex] File size: %.2f MB, Nodes: %u\n",
           file_size / (1024.0 * 1024.0), index->num_nodes);

    // Allocate host memory
    index->h_pIndex = (uint8_t*)malloc(file_size);
    if (!index->h_pIndex) {
        fprintf(stderr, "Failed to allocate host memory for static index\n");
        fclose(fp);
        exit(1);
    }

    // Read index file
    size_t read = fread(index->h_pIndex, 1, file_size, fp);
    if (read != file_size) {
        fprintf(stderr, "Error reading index file\n");
        free(index->h_pIndex);
        fclose(fp);
        exit(1);
    }
    fclose(fp);

    // Allocate device memory
    cudaError_t err = cudaMalloc(&index->d_pIndex, file_size);
    gpuErrchk(err);

    // Copy to device
    err = cudaMemcpy(index->d_pIndex, index->h_pIndex, file_size, cudaMemcpyHostToDevice);
    gpuErrchk(err);

    printf("[StaticIndex] Loaded %u nodes to GPU\n", index->num_nodes);
}

bool shouldConsolidate(FreshIndex* fresh, DeleteBuffer* del_buf, double elapsed_time) {
    uint32_t fresh_size = *fresh->h_count;

    // Check size threshold
    bool size_trigger = (fresh_size >= CONSOLIDATE_SIZE_THRESHOLD);

    // Check time threshold
    bool time_trigger = (elapsed_time >= CONSOLIDATE_TIME_THRESHOLD);

    if (size_trigger || time_trigger) {
        printf("[Consolidation] Triggered: fresh_size=%u/%u (%.1f%%), elapsed=%.1fs\n",
               fresh_size, fresh->capacity,
               100.0 * fresh_size / fresh->capacity,
               elapsed_time);
        return true;
    }

    return false;
}

void consolidateIndices(StaticIndex* static_idx, FreshIndex* fresh, DeleteBuffer* del_buf) {
    printf("[Consolidation] Starting consolidation...\n");

    CPUTimer timer;
    timer.Start();

    // Step 1: Copy current fresh index to host
    uint32_t fresh_size = *fresh->h_count;
    cudaError_t err = cudaMemcpy(fresh->h_pIndex, fresh->d_pIndex,
                                 fresh->total_size_bytes, cudaMemcpyDeviceToHost);
    gpuErrchk(err);

    // Step 2: Count active nodes (non-deleted)
    uint32_t static_active = static_idx->num_nodes - del_buf->num_deleted;
    uint32_t fresh_active = fresh_size;  // Fresh nodes are not in delete buffer yet
    uint32_t total_active = static_active + fresh_active;

    printf("[Consolidation] Active nodes: static=%u, fresh=%u, total=%u\n",
           static_active, fresh_active, total_active);

    // Step 3: Allocate new combined index on host
    size_t new_size_bytes = total_active * INDEX_ENTRY_LEN;
    uint8_t* h_new_index = (uint8_t*)malloc(new_size_bytes);
    if (!h_new_index) {
        fprintf(stderr, "Failed to allocate memory for consolidated index\n");
        exit(1);
    }

    // Step 4: Copy active nodes from static index
    uint32_t write_pos = 0;
    for (uint32_t i = 0; i < static_idx->num_nodes; i++) {
        if (!isNodeDeleted(del_buf, i)) {
            memcpy(h_new_index + write_pos * INDEX_ENTRY_LEN,
                   static_idx->h_pIndex + i * INDEX_ENTRY_LEN,
                   INDEX_ENTRY_LEN);
            write_pos++;
        }
    }

    // Step 5: Copy all nodes from fresh index
    for (uint32_t i = 0; i < fresh_size; i++) {
        memcpy(h_new_index + write_pos * INDEX_ENTRY_LEN,
               fresh->h_pIndex + i * INDEX_ENTRY_LEN,
               INDEX_ENTRY_LEN);
        write_pos++;
    }

    printf("[Consolidation] Copied %u active nodes\n", write_pos);

    // Step 6: TODO: Rebuild graph using Vamana algorithm
    // For now, we just keep existing edges (simplified)
    // Full implementation would call BANG-Variants-vamana-gpu here

    // Step 7: Free old static index
    if (static_idx->d_pIndex) {
        cudaFree(static_idx->d_pIndex);
    }
    if (static_idx->h_pIndex) {
        free(static_idx->h_pIndex);
    }

    // Step 8: Allocate new device memory and copy
    err = cudaMalloc(&static_idx->d_pIndex, new_size_bytes);
    gpuErrchk(err);

    err = cudaMemcpy(static_idx->d_pIndex, h_new_index, new_size_bytes, cudaMemcpyHostToDevice);
    gpuErrchk(err);

    // Step 9: Update static index metadata
    static_idx->h_pIndex = h_new_index;
    static_idx->num_nodes = total_active;
    static_idx->total_size_bytes = new_size_bytes;

    // Step 10: Clear fresh index and delete buffer
    clearFreshIndex(fresh);
    clearDeleteBuffer(del_buf);

    timer.Stop();
    printf("[Consolidation] Completed in %.2f seconds\n", timer.Elapsed());
    printf("[Consolidation] New static index: %u nodes, %.2f MB\n",
           static_idx->num_nodes,
           new_size_bytes / (1024.0 * 1024.0));
}

void freeStaticIndex(StaticIndex* index) {
    if (index->d_pIndex) {
        cudaFree(index->d_pIndex);
        index->d_pIndex = nullptr;
    }
    if (index->h_pIndex) {
        free(index->h_pIndex);
        index->h_pIndex = nullptr;
    }
    index->num_nodes = 0;
    index->capacity = 0;
}

void printIndexStats(const StaticIndex* static_idx, const FreshIndex* fresh_idx,
                     const DeleteBuffer* del_buf) {
    uint32_t fresh_size = *fresh_idx->h_count;
    uint32_t total_nodes = static_idx->num_nodes + fresh_size;
    uint32_t active_nodes = total_nodes - del_buf->num_deleted;

    printf("\n========== Index Statistics ==========\n");
    printf("Static Index:  %u nodes (%.2f MB)\n",
           static_idx->num_nodes,
           static_idx->total_size_bytes / (1024.0 * 1024.0));
    printf("Fresh Index:   %u / %u nodes (%.1f%% full)\n",
           fresh_size, fresh_idx->capacity,
           100.0 * fresh_size / fresh_idx->capacity);
    printf("Deleted:       %u nodes (%.1f%% of total)\n",
           del_buf->num_deleted,
           100.0 * del_buf->num_deleted / total_nodes);
    printf("Active Nodes:  %u\n", active_nodes);
    printf("======================================\n\n");
}
