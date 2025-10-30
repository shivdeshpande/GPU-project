#ifndef FILE_LOADERS_H
#define FILE_LOADERS_H

#include <fstream>
#include <iostream>
#include <cstring>
#include <cstdint>
#include "dynamicBANG.h"

// Copied from BANG_Exactdistance parANN.h

class cached_ifstream {
public:
    cached_ifstream() {}
    cached_ifstream(const std::string& filename, uint64_t cacheSize) {
        this->open(filename, cacheSize);
    }
    ~cached_ifstream() {
        if (cache_buf) delete[] cache_buf;
        reader.close();
    }

    void open(const std::string& filename, uint64_t cacheSize) {
        this->cur_off = 0;
        reader.open(filename, std::ios::binary | std::ios::ate);
        fsize = reader.tellg();
        reader.seekg(0, std::ios::beg);
        assert(reader.is_open());
        assert(cacheSize > 0);
        cacheSize = (std::min)(cacheSize, fsize);
        this->cache_size = cacheSize;
        cache_buf = new char[cacheSize];
        reader.read(cache_buf, cacheSize);
    }

    size_t get_file_size() { return fsize; }

    void read(char* read_buf, uint64_t n_bytes) {
        assert(cache_buf != nullptr);
        assert(read_buf != nullptr);
        if (n_bytes <= (cache_size - cur_off)) {
            memcpy(read_buf, cache_buf + cur_off, n_bytes);
            cur_off += n_bytes;
        } else {
            uint64_t cached_bytes = cache_size - cur_off;
            memcpy(read_buf, cache_buf + cur_off, cached_bytes);
            reader.read(read_buf + cached_bytes, n_bytes - cached_bytes);
            cur_off = cache_size;

            uint64_t size_left = fsize - reader.tellg();
            if (size_left >= cache_size) {
                reader.read(cache_buf, cache_size);
                cur_off = 0;
            }
        }
    }

private:
    std::ifstream reader;
    uint64_t cache_size = 0;
    char* cache_buf = nullptr;
    uint64_t cur_off = 0;
    uint64_t fsize = 0;
};

inline void load_truthset(const std::string& bin_file, uint32_t*& ids,
                          float*& dists, size_t& npts, size_t& dim) {
    // Support both .ivecs format and custom binary format
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    size_t actual_file_size = reader.tellg();
    reader.seekg(0, std::ios::beg);

    // Try .ivecs format first (each vector: [dim][val1][val2]...[valdim])
    int first_dim;
    reader.read((char*)&first_dim, sizeof(int));

    // Check if this looks like .ivecs format
    size_t record_size = sizeof(int) + first_dim * sizeof(uint32_t);
    if (actual_file_size % record_size == 0) {
        // .ivecs format detected
        npts = actual_file_size / record_size;
        dim = first_dim;

        printf("[Truthset] Loading .ivecs: #pts = %lu, #dims = %lu\n", npts, dim);

        ids = new uint32_t[npts * dim];
        dists = nullptr;  // .ivecs doesn't include distances

        reader.seekg(0, std::ios::beg);
        for (size_t i = 0; i < npts; i++) {
            int d;
            reader.read((char*)&d, sizeof(int));
            if (d != (int)dim) {
                fprintf(stderr, "Error: Dimension mismatch at vector %lu\n", i);
                break;
            }
            reader.read((char*)(ids + i * dim), dim * sizeof(uint32_t));
        }
    } else {
        // Custom binary format: [npts][dim][ids...][dists...]
        reader.seekg(0, std::ios::beg);
        int npts_i32, dim_i32;
        reader.read((char*)&npts_i32, sizeof(int));
        reader.read((char*)&dim_i32, sizeof(int));
        npts = (unsigned)npts_i32;
        dim = (unsigned)dim_i32;

        printf("[Truthset] Loading custom format: #pts = %lu, #dims = %lu\n", npts, dim);

        size_t expected_file_size = 2 * npts * dim * sizeof(uint32_t) + 2 * sizeof(uint32_t);
        if (actual_file_size != expected_file_size) {
            fprintf(stderr, "Warning: File size mismatch. Actual=%lu, Expected=%lu\n",
                    actual_file_size, expected_file_size);
        }

        ids = new uint32_t[npts * dim];
        reader.read((char*)ids, npts * dim * sizeof(uint32_t));
        dists = new float[npts * dim];
        reader.read((char*)dists, npts * dim * sizeof(float));
    }

    reader.close();
}

template<typename T>
inline void load_aligned_bin(const std::string& bin_file, T*& data,
                             size_t& npts, size_t& dim, size_t& rounded_dim) {
    std::ifstream reader(bin_file, std::ios::binary | std::ios::ate);
    uint64_t fsize = reader.tellg();
    reader.seekg(0);

    int npts_i32, dim_i32;
    reader.read((char*)&npts_i32, sizeof(int));
    reader.read((char*)&dim_i32, sizeof(int));
    npts = (unsigned)npts_i32;
    dim = (unsigned)dim_i32;

    rounded_dim = ROUND_UP(dim, 8);

    size_t allocSize = npts * rounded_dim * sizeof(T);
    data = (T*)malloc(allocSize);
    assert(data != nullptr);

    for (size_t i = 0; i < npts; i++) {
        reader.read((char*)(data + i * rounded_dim), dim * sizeof(T));
        memset(data + i * rounded_dim + dim, 0, (rounded_dim - dim) * sizeof(T));
    }

    reader.close();
    printf("[BinFile] Loaded %lu vectors of dimension %lu (rounded to %lu)\n",
           npts, dim, rounded_dim);
}

#endif // FILE_LOADERS_H
