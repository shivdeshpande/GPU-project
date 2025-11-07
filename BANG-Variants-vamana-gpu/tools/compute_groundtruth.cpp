#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cfloat>

/**
 * Compute brute-force groundtruth for ANNS
 * Computes exact k-nearest neighbors using exhaustive L2 distance
 */

struct DistanceIdPair {
    float distance;
    unsigned id;

    bool operator<(const DistanceIdPair& other) const {
        return distance < other.distance;
    }
};

/**
 * Compute L2 distance between two vectors
 */
float l2Distance(const float* vec1, const float* vec2, unsigned dim) {
    float sum = 0.0f;
    for (unsigned i = 0; i < dim; i++) {
        float diff = vec1[i] - vec2[i];
        sum += diff * diff;
    }
    return sum; // Return squared distance (no sqrt needed for comparison)
}

/**
 * Load binary file in format: [num_points][dim][vector_data...]
 */
bool loadBinaryFile(const char* filename, float** data, unsigned* numPoints, unsigned* dim) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file: " << filename << std::endl;
        return false;
    }

    unsigned np, d;
    file.read((char*)&np, sizeof(unsigned));
    file.read((char*)&d, sizeof(unsigned));

    *numPoints = np;
    *dim = d;

    std::cout << "Loading " << filename << ": " << np << " points, " << d << " dimensions" << std::endl;

    *data = new float[np * d];
    file.read((char*)(*data), np * d * sizeof(float));

    file.close();
    return true;
}

/**
 * Save groundtruth in format: [num_queries][k][id1][id2]...[idk]
 */
bool saveGroundtruth(const char* filename, const std::vector<std::vector<unsigned>>& groundtruth) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filename << std::endl;
        return false;
    }

    unsigned numQueries = groundtruth.size();
    unsigned k = groundtruth[0].size();

    file.write((char*)&numQueries, sizeof(unsigned));
    file.write((char*)&k, sizeof(unsigned));

    for (unsigned i = 0; i < numQueries; i++) {
        for (unsigned j = 0; j < k; j++) {
            unsigned id = groundtruth[i][j];
            file.write((char*)&id, sizeof(unsigned));
        }
    }

    file.close();
    std::cout << "Groundtruth saved to " << filename << std::endl;
    std::cout << "  " << numQueries << " queries, k=" << k << std::endl;

    return true;
}

/**
 * Compute groundtruth for all queries
 */
void computeGroundtruth(const float* baseData, unsigned numBase, unsigned dim,
                        const float* queryData, unsigned numQueries,
                        unsigned k, std::vector<std::vector<unsigned>>& groundtruth) {

    groundtruth.resize(numQueries);

    std::cout << "\nComputing brute-force groundtruth..." << std::endl;

    for (unsigned q = 0; q < numQueries; q++) {
        if ((q + 1) % 10 == 0 || q == 0) {
            std::cout << "  Processing query " << (q + 1) << "/" << numQueries << "..." << std::endl;
        }

        const float* query = queryData + q * dim;

        // Compute distances to all base points
        std::vector<DistanceIdPair> distances;
        distances.reserve(numBase);

        for (unsigned i = 0; i < numBase; i++) {
            const float* basePoint = baseData + i * dim;
            float dist = l2Distance(query, basePoint, dim);
            distances.push_back({dist, i});
        }

        // Sort by distance
        std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

        // Extract top-k IDs
        groundtruth[q].resize(k);
        for (unsigned i = 0; i < k; i++) {
            groundtruth[q][i] = distances[i].id;
        }
    }

    std::cout << "Groundtruth computation complete!" << std::endl;
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cout << "Usage: " << argv[0] << " <base_file> <query_file> <k> <output_file>" << std::endl;
        std::cout << "\nComputes brute-force groundtruth using exhaustive L2 distance" << std::endl;
        std::cout << "\nExample:" << std::endl;
        std::cout << "  " << argv[0] << " data/base.bin data/siftsmall_query.bin 100 data/groundtruth_bruteforce.bin" << std::endl;
        return 1;
    }

    const char* baseFile = argv[1];
    const char* queryFile = argv[2];
    unsigned k = atoi(argv[3]);
    const char* outputFile = argv[4];

    std::cout << "╔════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Brute-Force Groundtruth Computation     ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════╝\n" << std::endl;

    // Load base dataset
    float* baseData;
    unsigned numBase, baseDim;
    if (!loadBinaryFile(baseFile, &baseData, &numBase, &baseDim)) {
        return 1;
    }

    // Load query dataset
    float* queryData;
    unsigned numQueries, queryDim;
    if (!loadBinaryFile(queryFile, &queryData, &numQueries, &queryDim)) {
        delete[] baseData;
        return 1;
    }

    // Verify dimensions match
    if (baseDim != queryDim) {
        std::cerr << "Error: Dimension mismatch! Base=" << baseDim << ", Query=" << queryDim << std::endl;
        delete[] baseData;
        delete[] queryData;
        return 1;
    }

    if (k > numBase) {
        std::cerr << "Error: k (" << k << ") cannot be greater than number of base points (" << numBase << ")" << std::endl;
        delete[] baseData;
        delete[] queryData;
        return 1;
    }

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Base points: " << numBase << std::endl;
    std::cout << "  Query points: " << numQueries << std::endl;
    std::cout << "  Dimensions: " << baseDim << std::endl;
    std::cout << "  k (neighbors): " << k << std::endl;

    // Compute groundtruth
    std::vector<std::vector<unsigned>> groundtruth;
    computeGroundtruth(baseData, numBase, baseDim, queryData, numQueries, k, groundtruth);

    // Save groundtruth
    if (!saveGroundtruth(outputFile, groundtruth)) {
        delete[] baseData;
        delete[] queryData;
        return 1;
    }

    std::cout << "\n✓ Success! Groundtruth saved to: " << outputFile << std::endl;

    delete[] baseData;
    delete[] queryData;

    return 0;
}
