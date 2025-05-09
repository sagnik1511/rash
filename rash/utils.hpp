#pragma once

#include <iostream>
#include <numeric>
#include <vector>

/**
 * @brief Generates a range of integers from startIndex to endIndex.
 */
std::vector<int> arange(int startIndex, int endIndex) {
    std::vector<int> v(endIndex - startIndex, 0);
    std::iota(v.begin(), v.end(), startIndex);

    return v;
}

/**
 * @brief Prints a vector of integers.
 */
void printVec(const std::vector<int>& vec) {
    std::cout << "[ ";
    for (int i = 0; i < vec.size() - 1; i++) {
        std::cout << vec[i] << ", ";
    }
    std::cout << vec[vec.size() - 1] << " ]\n";
}