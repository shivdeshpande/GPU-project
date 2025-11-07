#!/usr/bin/env python3
"""
Convert .ivecs groundtruth file to .bin format

.ivecs format: For each vector: [dim:4bytes][val1:4bytes][val2:4bytes]...
.bin format: [num_vectors:4bytes][dim:4bytes][all_values:4bytes_each]
"""

import struct
import sys
import numpy as np

def read_ivecs(filename):
    """Read .ivecs file"""
    print(f"Reading {filename}...")

    vectors = []
    with open(filename, 'rb') as f:
        while True:
            # Read dimension
            dim_bytes = f.read(4)
            if not dim_bytes:
                break

            dim = struct.unpack('i', dim_bytes)[0]

            # Read vector values
            vector = struct.unpack(f'{dim}i', f.read(4 * dim))
            vectors.append(vector)

    vectors_array = np.array(vectors, dtype=np.uint32)
    print(f"  Loaded {vectors_array.shape[0]} vectors, dim={vectors_array.shape[1]}")

    return vectors_array

def write_bin(filename, data):
    """Write .bin file in format: [num][dim][data...]"""
    print(f"Writing {filename}...")

    num_vectors = data.shape[0]
    dim = data.shape[1]

    with open(filename, 'wb') as f:
        # Write header
        f.write(struct.pack('I', num_vectors))
        f.write(struct.pack('I', dim))

        # Write all data
        f.write(data.tobytes())

    print(f"  Wrote {num_vectors} vectors, dim={dim}")
    print(f"  File size: {(num_vectors * dim * 4 + 8) / 1024:.2f} KB")

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_ivecs_to_bin.py <input.ivecs> <output.bin>")
        print("\nExample:")
        print("  python convert_ivecs_to_bin.py siftsmall_groundtruth.ivecs groundtruth.bin")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    print("╔════════════════════════════════════════════╗")
    print("║   .ivecs to .bin Converter                ║")
    print("╚════════════════════════════════════════════╝\n")

    # Read .ivecs file
    data = read_ivecs(input_file)

    # Write .bin file
    write_bin(output_file, data)

    print(f"\n✓ Conversion complete!")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    main()
