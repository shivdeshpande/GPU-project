#include <bits/stdc++.h>
using namespace std;
#include <cuvs/neighbors/vamana.hpp>

#define N 1000000
#define R 64
#define D 128

int main(void) {
    // Read the graph into vectors
    std::ifstream fin("~/varun/sift1m_index_disk.bin", std::ios::binary);
    std::vector<float> vectors(N * D);
    std::vector<uint32_t> graph(N * R);
    for (int i = 0; i < N; ++i) {
        fin.read(reinterpret_cast<char*>(&vectors[i * D]), sizeof(float) * D);
        fin.seekg(sizeof(uint32_t), std::ios::cur);
        fin.read(reinterpret_cast<char*>(&graph[i * R]), sizeof(float) * D);
    }
    fin.close();

    
    // Copy innput to GPU
    raft::resources handle;
    auto d_vectors = raft::make_device_matrix<float>(handle, N, D);
    raft::copy(d_vectors.data_handle(), vectors.data(), N * D, raft::resource::get_cuda_stream(handle));
    
    auto d_graph = raft::make_device_matrix<uint32_t>(handle, N, D);
    raft::copy(d_graph.data_handle(), graph.data(), N * R, raft::resource::get_cuda_stream(handle));

    // Vamana
    cuvs::neighbors::vamana::index_params index_params;
    index_params.metric = cuvs::distance::DistanceType::L2Expanded;
    index_params.graph_degree = 64;
    index_params.visited_size = 512;

    auto vecs_view = raft::make_const_mdspan(d_vectors.view());
    auto index = cuvs::neighbors::vamana::build(handle, index_params, vecs_view);

    // Copt result to CPU
    std::vector<uint32_t> host_graph(N * R);
    std::vector<float> host_vectors(N * D);
    raft::copy(host_vectors.data(), d_vectors.data_handle(), N * D, raft::resource::get_cuda_stream(handle));
    raft::copy(host_graph.data(), index.graph().data_handle(), N * R, raft::resource::get_cuda_stream(handle));

    cout << "Medoid: " << index.medoid() << endl;

    // Write to combined output file
    std::ofstream fout("sift1m.out", std::ios::binary);
    for (int i = 0; i < N; ++i) {
        fout.write(reinterpret_cast<const char*>(&host_vectors[i * D]), sizeof(float) * D);

        uint32_t deg = 0;
        for (int j = 0; j < R; ++j) {
            uint32_t neighbor = static_cast<uint32_t>(host_graph[i * R + j]);
            if (neighbor != 4294967295UL) {
                deg++;
            } else {
                break;
            }
        }

        fout.write(reinterpret_cast<const char*>(&deg), sizeof(uint32_t));

        for (int j = 0; j < R; ++j) {
            uint32_t neighbor = static_cast<uint32_t>(host_graph[i * R + j]);
            if (j >= deg) neighbor = 0;
            fout.write(reinterpret_cast<const char*>(&neighbor), sizeof(uint32_t));
        }
    }
    fout.close();


    cout << "Done" << endl;
    return 0;
}
