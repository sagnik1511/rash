// g++ --std=c++17 tests/matmul.cpp -o matmul -lopenblas -g
#include <iostream>

#include "../rash/tensorMeta.hpp"

int main() {
    TensorMeta meta = TensorMeta({3});
    TensorMeta newMeta = TensorMeta({1, 3, 4, 3, 1});

    TensorMeta out = TensorMeta::matmul(meta, newMeta);
    out.shape();
    out.display();

    return 0;
}
