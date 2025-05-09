#pragma once
#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

#include "utils.hpp"

#define DOUBLE_MAX std::numeric_limits<double>::infinity()

namespace rash {

/**
 * @class TensorMeta
 * @brief A class to represent tensor data with various operations and transformations.
 */
class TensorMeta {
    int numel;
    std::vector<int> tensorSize;

   public:
    std::vector<double> rawData;

#pragma region DataDefinition
    /**
     * @brief Constructs a TensorMeta object with given data and shape.
     * @param data The raw data.
     * @param size The shape of the tensor.
     * @throws std::runtime_error if the data size does not match the tensor shape.
     */
    TensorMeta(std::vector<double> data, std::vector<int> size) : tensorSize(size), rawData(data) {
        numel = 1;
        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        if (rawData.size() != numel) {
            throw std::runtime_error("Data size mismatch with tensorSize!");
        }
    }

    /**
     * @brief Constructs a scalar TensorMeta object.
     * @param data The scalar value.
     */
    TensorMeta(double data) : tensorSize({1}), rawData({data}) { numel = 1; }

    /**
     * @brief Constructs a TensorMeta object with a given shape and initializes it with random values.
     * @param size The shape of the tensor.
     */
    TensorMeta(std::vector<int> size) : tensorSize(size) {
        numel = 1;

        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        rawData.assign(numel, 0.0);
    }
    /**
     * @brief Default constructor for TensorMeta.
     */
    TensorMeta() = default;

    /**
     * @brief Default destructor for TensorMeta.
     */
    ~TensorMeta() = default;

    /**
     * @brief Copy constructor for TensorMeta.
     */
    TensorMeta(const TensorMeta& other) : numel(other.numel), tensorSize(other.tensorSize), rawData(other.rawData) {}

    /**
     * @brief Fills the tensor with random values between 0 and 1.
     */
    void fillRandomData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < numel; i++) {
            rawData[i] = dis(gen);
        }
    }

    /**
     * @brief Updates all tensor elements to a specified value.
     * @param value The new value for all elements.
     */
    void updateAll(double value) { rawData.assign(numel, value); }

    /**
     * @brief Shows the tensor data in a formatted manner.
     * @param os The output stream.
     * @param shape The shape of the tensor.
     * @param flattenedData The flattened data of the tensor.
     * @param startIdx The starting index for displaying.
     */
    static void showRecursive(std::ostream& os, std::vector<int> shape, const std::vector<double>& flattenedData,
                              int startIdx = 0) {
        if (shape.size() == 1) {
            os << "[";
            for (int i = startIdx; i < startIdx + shape[0] - 1; i++) {
                os << flattenedData[i] << " ,";
            }
            os << flattenedData[startIdx + shape[0] - 1] << "]";
        } else {
            std::vector<int> childShape(shape.begin() + 1, shape.end());
            int totalElPerDim = 1;
            for (auto el : childShape) {
                totalElPerDim *= el;
            }
            os << "[";
            for (int i = 0; i < shape[0] - 1; i++) {
                showRecursive(os, childShape, flattenedData, startIdx + (i * totalElPerDim));
                os << ", \n";
            }
            showRecursive(os, childShape, flattenedData, startIdx + ((shape[0] - 1) * totalElPerDim));
            os << "]";
        }
    }

    /**
     * @brief Displays the tensor in a formatted manner.
     */
    static void display(std::ostream& oss, const TensorMeta& meta) {
        showRecursive(oss, meta.tensorSize, meta.rawData, 0);
    }

    /**
     * @brief Overloads the output stream operator for printing tensors.
     */
    friend std::ostream& operator<<(std::ostream& os, const TensorMeta& meta) {
        meta.display(os, meta);
        return os;
    }

#pragma endregion

    /**
     * @brief Removes singleton dimensions from the tensor.
     * @param dims The dimensions to be squeezed.
     * @return A new TensorMeta object with squeezed dimensions.
     */
    TensorMeta squeeze(std::vector<int> dims) const {
        std::vector<int> newSize(tensorSize);
        std::sort(dims.begin(), dims.end(), std::greater<>());
        for (auto& idx : dims) {
            if (idx < ndim() && tensorSize[idx] == 1)
                newSize.erase(newSize.begin() + idx);
        }

        return TensorMeta(rawData, newSize);
    }

    /**
     * @brief Removes a singleton dimension.
     * @param dim The dimension to be squeezed.
     * @return A new TensorMeta object.
     */
    TensorMeta squeeze(int dim = 0) const {
        std::vector<int> dims = {dim};
        return squeeze(dims);
    }

    /**
     * @brief Expands the tensor by inserting a singleton dimension.
     * @param dim The position of the new dimension.
     * @return A new TensorMeta object.
     */
    TensorMeta unsqueeze(int dim = 0) const {
        std::vector<int> newSize(tensorSize);
        newSize.insert(newSize.begin() + dim, 1);
        return TensorMeta(rawData, newSize);
    }

    /**
     * @brief Computes the broadcasted shape for two tensors.
     * @param sz1 The shape of the first tensor.
     * @param sz2 The shape of the second tensor.
     * @return The broadcasted shape.
     * @throws std::length_error if broadcasting is not possible.
     */
    static std::vector<int> fetchBroadcastedSize(const std::vector<int>& sz1, const std::vector<int>& sz2) {
        if (!sz1.size() || !sz2.size()) {
            throw std::length_error("Tensor should have atleats a dimension!");
            std::exit(-1);
        }

        std::vector<int> curr(sz1.rbegin(), sz1.rend()), incm(sz2.rbegin(), sz2.rend());

        int idx = 0;
        int n = curr.size(), m = incm.size();
        std::vector<int> finSize;
        while (idx < n || idx < m) {
            if (idx < n && idx < m) {
                if (curr[idx] == 0 || incm[idx] == 0) {
                    throw std::runtime_error("Size mismatch in Broadcasting");
                    std::exit(-1);
                } else if (curr[idx] == incm[idx]) {
                    finSize.push_back(curr[idx]);
                } else if (curr[idx] == 1) {
                    finSize.push_back(incm[idx]);
                } else if (incm[idx] == 1) {
                    finSize.push_back(curr[idx]);
                } else {
                    throw std::length_error("Size mismatch in Broadcasting");
                }
            } else if (idx < n) {
                finSize.push_back(curr[idx]);
            } else if (idx < m) {
                finSize.push_back(incm[idx]);
            }
            idx++;
        }
        std::reverse(finSize.begin(), finSize.end());

        return finSize;
    }

    /**
     * @brief Computes the broadcasted shape for two tensors.
     * @param dat1 The first tensor.
     * @param dat2 The second tensor.
     * @return The broadcasted shape.
     */
    static std::vector<int> fetchBroadcastedSize(const TensorMeta& dat1, const TensorMeta& dat2) {
        return fetchBroadcastedSize(dat1.tensorSize, dat2.tensorSize);
    }

    static std::vector<int> fetchStride(const TensorMeta& data) { return fetchStride(data.tensorSize); }

    static std::vector<int> fetchStride(const std::vector<int>& shape) {
        int currDimStride = 1;
        std::vector<int> stride(shape.size(), 0);
        for (int idx = shape.size() - 1; idx >= 0; idx--) {
            stride[idx] = currDimStride;
            currDimStride *= shape[idx];
        }

        return stride;
    }

    /**
     * @brief Computes the flat index for given indices in a tensor with a given shape and stride.
     * @param indices The indices in multi-dimensional space.
     * @param shape The shape of the tensor.
     * @param stride The stride of the tensor.
     * @return The computed flat index.
     */
    static int getIndex(const std::vector<int>& indices, const std::vector<int>& shape,
                        const std::vector<int>& stride) {
        int idx = 0;
        int dimOffset = indices.size() - shape.size();

        for (size_t i = 0; i < shape.size(); ++i) {
            idx += (shape[i] == 1) ? 0 : indices[i + dimOffset] * stride[i];
        }

        return idx;
    }

    /**
     * @brief Performs element-wise broadcasting operation on two tensors.
     * @param dat1 The first tensor.
     * @param dat2 The second tensor.
     * @param op The operation function.
     * @return The resulting tensor after applying the operation.
     */
    static TensorMeta broadcast(const TensorMeta& dat1, const TensorMeta& dat2,
                                std::function<double(double, double)> op) {
        TensorMeta out(fetchBroadcastedSize(dat1, dat2));

        std::vector<int> indices(out.ndim(), 0);

        std::vector<int> stride1 = fetchStride(dat1);
        std::vector<int> stride2 = fetchStride(dat2);
        std::vector<int> strideOut = fetchStride(out);

        for (int idx = 0; idx < out.numel; ++idx) {
            int idx1 = getIndex(indices, dat1.tensorSize, stride1);
            int idx2 = getIndex(indices, dat2.tensorSize, stride2);

            out.rawData[idx] = op(dat1.rawData[idx1], dat2.rawData[idx2]);

            // Update indices for broadcasting
            for (int dim = indices.size() - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < out.tensorSize[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

        return out;
    }

    /**
     * @brief Returns the number of dimensions of the tensor.
     * @return The number of dimensions.
     */
    int ndim() const { return tensorSize.size(); }

    /**
     * @brief Returns the shape of the tensor.
     * @return The shape of the tensor.
     */
    std::vector<int> shape() { return tensorSize; }

#pragma region ArithMaticOperators

    /**
     * @brief Computes the element-wise/broadcastabed addition of two tensors.
     * @param other The other tensor.
     * @return A new tensor with the result of the addition.
     */
    TensorMeta operator+(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed addition of a tensor and a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the addition.
     */
    TensorMeta operator+(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        TensorMeta otherMeta = TensorMeta({other}, {1});
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed addition of two tensors and assigns the result to the first
     * tensor.
     * @param other The other tensor.
     * @return A reference to the current tensor.
     * @throws std::runtime_error if the shapes are not compatible.
     */
    TensorMeta& operator+=(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        *this = std::move(TensorMeta::broadcast(*this, other, op));
        return *this;
    }

    /**
     * @brief Computes the element-wise/broadcastabed negation of the tensor.
     * @return A new tensor with the result of the negation.
     */
    TensorMeta operator-() {
        TensorMeta other({0}, {1});
        std::function<double(double, double)> op = [](double val1, double val2) { return val2 - val1; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed subtraction of two tensors.
     * @param other The other tensor.
     * @return A new tensor with the result of the subtraction.
     */
    TensorMeta operator-(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed subtraction of a tensor and a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the subtraction.
     * @throws std::runtime_error if the shapes are not compatible.
     */
    TensorMeta operator-(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        TensorMeta otherMeta = TensorMeta({other}, {1});
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed subtraction of two tensors and assigns the result to the first
     * tensor.
     * @param other The other tensor.
     * @return A reference to the current tensor.
     * @throws std::runtime_error if the shapes are not compatible.
     */
    TensorMeta& operator-=(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        *this = std::move(TensorMeta::broadcast(*this, other, op));
        return *this;
    }

    /**
     * @brief Computes the element-wise/broadcastabed multiplication of two tensors.
     * @param other The other tensor.
     * @return A new tensor with the result of the multiplication.
     */
    TensorMeta operator*(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed multiplication of a tensor and a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the multiplication.
     */
    TensorMeta operator*(const double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    TensorMeta operator/(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 / val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed/broadcastabed division of a tensor and a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the division.
     */
    TensorMeta operator/(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 / val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed greater than operation.
     * @param other The other tensor.
     * @return A new tensor with the result of the operation.
     * @throws std::runtime_error if the shapes are not compatible.
     */
    TensorMeta operator>(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 > val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed greater than operation with a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator>(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 > val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed greater than or equal to operation.
     * @param other The other tensor.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator>=(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 >= val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed greater than or equal to operation with a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator>=(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 >= val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed less than operation.
     * @param other The other tensor.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator<(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 < val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed less than operation with a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator<(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 < val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed less than or equal to operation.
     * @param other The other tensor.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator<=(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 <= val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    /**
     * @brief Computes element-wise/broadcastabed less than or equal to operation with a scalar.
     * @param other The scalar value.
     * @return A new tensor with the result of the operation.
     */
    TensorMeta operator<=(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 <= val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    static TensorMeta pow(const TensorMeta& meta, double power) {
        TensorMeta other(power);
        std::function<double(double, double)> op = [](double val1, double val2) { return std::pow(val1, val2); };
        return TensorMeta::broadcast(meta, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed exponential of the tensor.
     * @param meta The input tensor.
     * @return A tensor with exponentiated values.
     */
    static TensorMeta exp(const TensorMeta& meta) {
        TensorMeta other(1);
        std::function<double(double, double)> op = [](double val1, double val2) { return std::exp(val1); };
        return TensorMeta::broadcast(meta, other, op);
    }

    /**
     * @brief Computes the element-wise/broadcastabed absolute value of the tensor.
     * @param meta The input tensor.
     * @return A tensor with absolute values.
     */
    static TensorMeta abs(const TensorMeta& meta) {
        TensorMeta other(1);
        std::function<double(double, double)> op = [](double val1, double val2) { return std::abs(val1); };
        return TensorMeta::broadcast(meta, other, op);
    }

    /**
     * @brief Converts a scalar tensor to a double.
     * @return The scalar value.
     * @throws std::runtime_error if the tensor is not a scalar.
     */
    operator double() const {
        if (ndim() == 1 && tensorSize[0] == 1) {
            return rawData[0];
        } else {
            throw std::runtime_error("Higher Dimensional data can't be converted to Scalar-type");
        }
    }

#pragma endregion

#pragma region MatMul
    /**
     * @brief Validates if two tensors can be multiplied using matrix multiplication.
     * @param dat1 The first tensor.
     * @param dat2 The second tensor.
     * @return True if matrix multiplication is valid, false otherwise.
     */
    static bool validateMatmul(const TensorMeta& dat1, const TensorMeta& dat2) {
        std::vector<int> v1, v2;
        int dim1 = dat1.ndim(), dim2 = dat2.ndim();
        v1 = dat1.tensorSize;
        v2 = dat2.tensorSize;

        if (dim1 == 1 && dim2 == 1) {
            return v1[0] == v2[0];
        } else if (dim1 == 1 && dim2 == 2) {
            // (K, ) x (K, N)
            return v1[0] == v2[0];
        } else if (dim1 == 2 && dim2 == 1) {
            // (M, K) x (K, )
            return v1[1] == v2[0];
        } else if (dim1 == 2 && dim2 == 2) {
            // (M, K) x (K, N)
            return v1[1] == v2[0];
        } else if (dim1 == 1 && dim2 > 2) {
            // Batched MatMul
            // (K, ) x (A, B, ..., K, N)
            return v1[0] == v2[dim2 - 2];
        } else if (dim1 > 2 && dim2 == 1) {
            // Batched MatMul
            // (A, B, ..., M, K) x (K, )
            return v1[dim1 - 1] == v2[0];
        } else {
            if (v1[dim1 - 1] != v2[dim2 - 2])
                return false;
            std::vector<int> v1Part(v1.begin(), v1.end() - 2), v2Part(v2.begin(), v2.end() - 2);
            try {
                if (!v1Part.size())
                    v1Part = {1};
                if (!v2Part.size())
                    v2Part = {1};
                // If broadcastable
                std::vector<int> bcSize = fetchBroadcastedSize(v1Part, v2Part);
                return true;
            } catch (const std::exception& e) {
                // If not broadcastable
                std::cerr << e.what() << std::endl;
                return false;
            }
        }
    }

    /**
     * @brief Computes the output shape for matrix multiplication, considering broadcasting rules.
     * @param dat1 First tensor metadata.
     * @param dat2 Second tensor metadata.
     * @return The shape of the resulting matrix after matmul.
     * @throws std::runtime_error If shapes are incompatible for matrix multiplication.
     */
    static std::vector<int> fetchMatmulSize(const TensorMeta& dat1, const TensorMeta& dat2) {
        bool execFlag = validateMatmul(dat1, dat2);
        if (!execFlag) {
            throw std::runtime_error("Shape mismatch for MatMul\n");
        }
        std::vector<int> sz1(dat1.tensorSize), sz2(dat2.tensorSize);
        int M = sz1[sz1.size() - 2], K = sz1[sz1.size() - 1], N = sz2[sz2.size() - 1];

        // If both Meta Storage are 2d Matrix
        if (sz1.size() == 2 && sz2.size() == 2) {
            return {M, N};
        }

        // Partition extra dim to fetch broadcasted size of batches
        std::vector<int> v1Part(sz1.begin(), sz1.end() - 2), v2Part(sz2.begin(), sz2.end() - 2);
        if (!v1Part.size())
            v1Part = {1};
        if (!v2Part.size())
            v2Part = {1};
        std::vector<int> matmulSize = fetchBroadcastedSize(v1Part, v2Part);

        // Add M, N dimension where
        // dat1 -> ..., M, K
        // dat2 -> ..., K, N
        matmulSize.push_back(M);
        matmulSize.push_back(N);

        return matmulSize;
    }

    /**
     * @brief Performs atomic matrix multiplication using BLAS (cblas_dgemm).
     * @param A First matrix (flattened vector).
     * @param B Second matrix (flattened vector).
     * @param out Output matrix (flattened vector).
     * @param offSetA Offset in A.
     * @param offSetB Offset in B.
     * @param offSetOut Offset in output matrix.
     * @param M Number of rows in A and output matrix.
     * @param K Number of columns in A and rows in B.
     * @param N Number of columns in B and output matrix.
     */
    static void matmulAtomic(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& out,
                             int offSetA, int offSetB, int offSetOut, int M, int K, int N) {
        // A ->   M x K
        // B ->   K x N
        // out -> M x N
        assert(A.size() >= offSetA + M * K && "A vector is too small!");
        assert(B.size() >= offSetB + K * N && "B vector is too small!");
        assert(out.size() >= offSetOut + M * N && "Output vector is too small!");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0, &A[offSetA], K, &B[offSetB], N, 0.0,
                    &out[offSetOut], N);
    }

    /**
     * @brief Computes the batch index offset for a given batch shape and strides.
     * @param shape Shape of the tensor.
     * @param stride Stride of the tensor.
     * @param indices Batch indices to compute offset.
     * @return The computed offset for the batch index.
     */
    static int getMatmulBatchIndex(const std::vector<int>& shape, const std::vector<int>& stride,
                                   const std::vector<int>& indices) {
        int offSet = 0;
        int dimShift = indices.size() - shape.size() + 2;
        // for(int idx=0;idx<shape.size()-2;++idx)
        for (int idx = 0; idx < shape.size() - 2; ++idx) {
            offSet += (shape[idx] == 1) ? 0 : indices[idx + dimShift] * stride[idx];
        }
        return offSet;
    }

    /**
     * @brief Performs batched matrix multiplication with broadcasting.
     * @param dat1 First tensor metadata.
     * @param dat2 Second tensor metadata.
     * @return The metadata of the resulting tensor after batched matrix multiplication.
     */
    static TensorMeta matmulBroadcast(const TensorMeta& dat1, const TensorMeta& dat2) {
        std::vector<int> outShape = fetchMatmulSize(dat1, dat2);
        TensorMeta out(outShape);

        int batchSize = 1;
        for (int dimIdx = 0; dimIdx < out.ndim() - 2; ++dimIdx) {
            batchSize *= out.tensorSize[dimIdx];
        }

        int M = outShape[outShape.size() - 2];
        int N = outShape[outShape.size() - 1];
        int K = dat1.tensorSize.back();

        std::vector<int> stride1 = fetchStride(dat1);
        std::vector<int> stride2 = fetchStride(dat2);
        std::vector<int> strideOut = fetchStride(out);

        std::vector<int> indices(outShape.size() - 2, 0);

        for (int batchIdx = 0; batchIdx < batchSize; ++batchIdx) {
            int offSet1 = getMatmulBatchIndex(dat1.tensorSize, stride1, indices);
            int offSet2 = getMatmulBatchIndex(dat2.tensorSize, stride2, indices);
            int offSetOut = batchIdx * (M * N);

            matmulAtomic(dat1.rawData, dat2.rawData, out.rawData, offSet1, offSet2, offSetOut, M, K, N);

            // Update indices for broadcasting
            for (int dim = indices.size() - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < outShape[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

        return out;
    }

    /**
     * @brief Computes the matrix multiplication of two tensors with broadcasting support.
     * @param dat1 First tensor metadata.
     * @param dat2 Second tensor metadata.
     * @return The resulting tensor metadata after matrix multiplication.
     * @throws std::runtime_error If matrix dimensions are inconsistent.
     */
    static TensorMeta matmul(const TensorMeta& dat1, const TensorMeta& dat2) {
        bool matmulFlag = validateMatmul(dat1, dat2);

        if (!matmulFlag)
            throw std::runtime_error("Inconsistent data dimension, unable to perform matmul!");

        int dim1 = dat1.ndim(), dim2 = dat2.ndim();

        if (dim1 == 1 && dim2 == 1) {
            // Performing Dot Product

            // A -> (M, ) , B -> (M, )
            // A -> (1, 1, M), B -> (1, M, 1)
            TensorMeta dat1Brodcasted = dat1.unsqueeze().unsqueeze();
            TensorMeta dat2Broadcasted = dat2.unsqueeze(1).unsqueeze();

            // Perform MatMul
            // Out -> (1, 1, 1)
            TensorMeta out = matmulBroadcast(dat1Brodcasted, dat2Broadcasted);

            // Out -> (1,)
            out = out.squeeze(2).squeeze();
            return out;
        } else if (dim1 == 2 && dim2 == 2) {
            // Expanding last dimension to perform batched matmul
            // A -> (M, K), B -> (K, N)
            // A -> (1, M, K), B -> (1, K, N)
            // Out -> (1, M, N)
            TensorMeta out = matmul(dat1.unsqueeze(), dat2.unsqueeze());

            // Out -> (M, N)
            return out.squeeze();
        } else if (dim1 == 1 && dim2 == 2) {
            // A -> (M, ) , B -> (M, K)
            // A -> (1, M), B -> (M, K)(unchanged)
            // Out -> (1, K)
            TensorMeta out = matmul(dat1.unsqueeze(), dat2);

            // Out -> (K,)
            return out.squeeze();
        } else if (dim1 == 2 && dim2 == 1) {
            // A -> (M, K) , B -> (K, )
            // A -> (M, K)(unchanged), B -> (K, 1)
            // Out -> (M, 1)
            TensorMeta out = matmul(dat1, dat2.unsqueeze(1));

            // Out -> (M,)
            return out.squeeze(1);
        } else {
            if (dim1 == 1) {
                // A -> (M, ) , B -> (..., M, K)
                // A -> (1, 1, M)
                TensorMeta dat1Brodcasted = dat1.unsqueeze().unsqueeze();
                // Out -> (..., 1, K)
                TensorMeta out = matmulBroadcast(dat1Brodcasted, dat2);
                // Out -> (..., K)
                return out.squeeze(out.ndim() - 2);
            } else if (dim1 == 2) {
                // A -> (M, K), B -> (..., K, N)
                // A -> (1, M, K)
                TensorMeta dat1Brodcasted = dat1.unsqueeze();
                // Out -> (..., M, N)
                return matmulBroadcast(dat1Brodcasted, dat2);
            } else if (dim2 == 1) {
                // A -> (..., M, K), B -> (K, )
                // B -> (1, K, 1)
                TensorMeta dat2Brodcasted = dat2.unsqueeze(1).unsqueeze();
                // Out -> (..., M, 1)
                TensorMeta out = matmulBroadcast(dat1, dat2Brodcasted);
                // Out -> (..., M)
                return out.squeeze(out.ndim() - 1);
            } else if (dim2 == 2) {
                // A -> (..., M, K), B -> (K, N)
                // B -> (1, K, N)
                TensorMeta dat2Brodcasted = dat2.unsqueeze();
                // Out -> (..., M, N)
                return matmulBroadcast(dat1, dat2Brodcasted);
            } else {
                return matmulBroadcast(dat1, dat2);
            }
        }
    }

#pragma endregion MatMul

#pragma region Reduce
    /**
     * @brief Computes the output shape after squeezing a tensor along given axes.
     * @param origShape Original shape of the tensor.
     * @param axis Axes along which to squeeze (optional).
     * @param keepdims If true, keeps dimensions as 1 instead of removing them.
     * @return The squeezed shape of the tensor.
     */
    static std::vector<int> fetchSqueezedShape(const std::vector<int>& origShape, std::vector<int> axis = {},
                                               bool keepdims = false) {
        if (!axis.size()) {
            return {1};
        }
        std::set<int> axes(axis.begin(), axis.end());
        std::vector<int> finShape(origShape.begin(), origShape.end());
        std::set<int>::reverse_iterator it;
        for (it = axes.rbegin(); it != axes.rend(); ++it) {
            int dim = *it;
            if (dim < origShape.size() && dim >= 0) {
                if (keepdims) {
                    finShape[dim] = 1;
                } else {
                    finShape.erase(finShape.begin() + dim);
                }
            }
        }
        if (finShape.empty())
            finShape = {1};

        return finShape;
    }

    /**
     * @brief Performs atomic summation while considering squeezed dimensions.
     * @param indices Indices of the base tensor.
     * @param baseMeta Input tensor data.
     * @param outMeta Output tensor data.
     * @param baseShape Shape of the input tensor.
     * @param baseStride Stride of the input tensor.
     * @param outShape Shape of the output tensor.
     * @param outStride Stride of the output tensor.
     * @param axis Axes along which to sum.
     * @param keepdims If true, retains reduced dimensions as size 1.
     */
    void squeezedSumAtomic(const std::vector<int>& indices, const std::vector<double>& baseMeta,
                           std::vector<double>& outMeta, const std::vector<int>& baseShape,
                           const std::vector<int>& baseStride, const std::vector<int>& outShape,
                           const std::vector<int>& outStride, const std::vector<int>& axis, bool keepdims = false) {
        // Fetch output indices
        std::vector<int> outIndices = indices;
        if (!keepdims)
            outIndices = fetchSqueezedShape(indices, axis);

        // Find actual indices in flattened data
        int baseIdx = getIndex(indices, baseShape, baseStride);
        int outIdx = getIndex(outIndices, outShape, outStride);

        //  In place Sum Operation
        outMeta[outIdx] += baseMeta[baseIdx];
    }

    /**
     * @brief Determines the broadcasted axes when expanding one tensor to match another.
     * @param base Original tensor metadata.
     * @param broadcasted Expanded tensor metadata.
     * @return A pair containing axes to broadcast and added dimensions.
     */
    static std::pair<std::vector<int>, std::vector<int>> fetchBroadcastedAxes(const TensorMeta& base,
                                                                              const TensorMeta& broadcasted) {
        std::vector<int> axes;
        std::vector<int> addedDims;
        int shift = broadcasted.ndim() - base.ndim();
        for (int i = 0; i < shift; ++i) addedDims.push_back(i);
        for (int i = 0; i < base.ndim(); ++i) {
            if (base.tensorSize[i] != broadcasted.tensorSize[i + shift]) {
                axes.push_back(i + shift);
            }
        }

        return {axes, addedDims};
    }

    /**
     * @brief Fetches reduction axis information for a given tensor and axis.
     *
     * This function computes the necessary values for reducing a tensor along a specified axis:
     *
     * - `jump`: The stride for accessing elements in the flattened array.
     *
     * - `numBatches`: The number of independent reductions that occur.
     *
     * - `incrementBatchIdx`: The step size to move between batches during reduction.
     *
     * - `outShape`: The shape of the output tensor after reduction.
     *
     * @param meta The metadata of the input tensor.
     * @param axis The axis along which reduction is performed.
     * @param keepDims If true, keeps the reduced dimension as size 1; otherwise, it is removed.
     * @return A tuple containing (jump, numBatches, incrementBatchIdx, outShape).
     */
    static std::tuple<int, int, int, std::vector<int>> fetchReduceAxInfo(const TensorMeta& meta, int axis,
                                                                         bool keepDims = false) {
        std::vector<int> outShape = fetchSqueezedShape(meta.tensorSize, {axis}, keepDims);
        int jump = 1;
        int numBatches = 1;
        for (int i = 0; i < meta.ndim(); i++) {
            if (i < axis)
                numBatches *= meta.tensorSize[i];
            if (i > axis)
                jump *= meta.tensorSize[i];
        }
        int incrementBatchIdx = jump * meta.tensorSize[axis];

        return std::make_tuple(jump, numBatches, incrementBatchIdx, outShape);
    }

    /**
     * @brief Performs reduction along multiple axes using a given operation.
     *
     * This function applies reduction sequentially over multiple axes, reducing one at a time.
     * The reduction function (e.g., sum, max, min) is applied to each axis in order.
     *
     * @param meta The input tensor metadata.
     * @param axis A list of axes along which to perform the reduction.
     * @param op A binary function defining the reduction operation.
     * @param keepDims If true, retains reduced dimensions as size 1.
     * @param initVal The initial value for the reduction operation.
     * @return A new TensorMeta object representing the reduced tensor.
     */
    static TensorMeta reduce(const TensorMeta& meta, std::vector<int> axis, std::function<double(double, double)> op,
                             bool keepDims = false, double initVal = 0) {
        // If no axis specified then perform on the whole data
        if (axis.size() == 0) {
            axis = arange(0, meta.ndim());
        }
        TensorMeta out = meta;
        sort(axis.rbegin(), axis.rend());
        for (auto dim : axis) {
            out = reduceSingle(out, dim, op, keepDims, initVal);
        }
        return out;
    }

    /**
     * @brief Performs reduction along a single axis.
     *
     * This function reduces the tensor along the specified axis using the provided binary operation.
     * It utilizes precomputed stride and batch size information for optimized indexing.
     *
     * @param meta The input tensor metadata.
     * @param ax The axis along which reduction is performed.
     * @param op A binary function defining the reduction operation (e.g., sum, max).
     * @param keepDims If true, retains the reduced dimension as size 1.
     * @param initVal The initial value for the reduction operation.
     * @return A new TensorMeta object representing the reduced tensor.
     */
    static TensorMeta reduceSingle(const TensorMeta& meta, int ax, std::function<double(double, double)> op,
                                   bool keepDims = false, double initVal = 0) {
        auto [jump, numBatches, incrementBatchIdx, outShape] = fetchReduceAxInfo(meta, ax, keepDims);
        TensorMeta out(outShape);
        out.updateAll(initVal);

        for (int idx = 0; idx < meta.numel; ++idx) {
            int outIdx = ((idx / incrementBatchIdx) * jump) + (idx % jump);
            out.rawData[outIdx] = op(out.rawData[outIdx], meta.rawData[idx]);
        }

        return out;
    }

    static TensorMeta sum(const TensorMeta& meta, std::vector<int> dims = {}, bool keepDims = false) {
        std::function<double(double, double)> op = [](double a, double b) { return a + b; };
        return reduce(meta, dims, op, keepDims);
    }

    static TensorMeta max(const TensorMeta& meta, std::vector<int> dims = {}, bool keepDims = false) {
        std::function<double(double, double)> op = [](double a, double b) { return std::max(a, b); };
        return reduce(meta, dims, op, keepDims);
    }

    static TensorMeta min(const TensorMeta& meta, std::vector<int> dims = {}, bool keepDims = false) {
        std::function<double(double, double)> op = [](double a, double b) { return std::min(a, b); };
        return reduce(meta, dims, op, keepDims, DOUBLE_MAX);
    }

    static TensorMeta mean(const TensorMeta& meta, std::vector<int> dims = {}, bool keepDims = false) {
        TensorMeta out = TensorMeta::sum(meta, dims, keepDims);
        int divisor = 1;
        for (auto ax : dims) {
            divisor *= meta.tensorSize[ax];
        }
        return out / double(divisor);
    }

    TensorMeta sum(std::vector<int> dims = {}, bool keepDims = false) { return sum(*this, dims, keepDims); }
    TensorMeta min(std::vector<int> dims = {}, bool keepDims = false) { return min(*this, dims, keepDims); }
    TensorMeta max(std::vector<int> dims = {}, bool keepDims = false) { return max(*this, dims, keepDims); }
    TensorMeta mean(std::vector<int> dims = {}, bool keepDims = false) { return mean(*this, dims, keepDims); }

#pragma endregion

#pragma region Wrangler
    /**
     * @brief Rearranges the dimensions of the tensor according to a given permutation.
     * @param perm The permutation order.
     * @return The permuted tensor metadata.
     * @throws std::runtime_error If permutation size does not match tensor dimensions.
     */
    TensorMeta permute(std::vector<int> perm) const {
        int n = ndim();
        assert(perm.size() == n && "Permutation Size Should Match with Original TensorMeta Size!");
        std::vector<int> indices(n, 0);
        std::vector<int> newShape(n, 0);
        std::vector<double> rawDataCopy(numel, -1);
        for (int dim = 0; dim < newShape.size(); dim++) {
            newShape[dim] = tensorSize[perm[dim]];
        }

        std::vector<int> stride = fetchStride(tensorSize);
        std::vector<int> newStride = fetchStride(newShape);

        for (int ix = 0; ix < numel; ++ix) {
            std::vector<int> newIndices(n, -1);

            // Fetch new multi indices
            for (int dim = 0; dim < newShape.size(); dim++) {
                newIndices[dim] = indices[perm[dim]];
            }
            // Prepare flattened index and assign to new memory chunk
            int newIndex = getIndex(newIndices, newShape, newStride);
            rawDataCopy[newIndex] = rawData[ix];

            for (int dim = n - 1; dim >= 0; dim--) {
                indices[dim]++;
                if (indices[dim] < tensorSize[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

        return TensorMeta(rawDataCopy, newShape);
    }

    /**
     * @brief Transposes two dimensions of the tensor.
     * @param dim1 First dimension to swap (default: -1, last dimension).
     * @param dim2 Second dimension to swap (default: -2, second last dimension).
     * @return The transposed tensor metadata.
     */
    TensorMeta transpose(int dim1 = -1, int dim2 = -2) {
        std::vector<int> perm = arange(0, ndim());
        // printVec(perm);
        if (dim1 < 0)
            dim1 = ndim() + dim1;
        if (dim2 < 0)
            dim2 = ndim() + dim2;

        std::iter_swap(perm.begin() + dim1, perm.begin() + dim2);

        return permute(perm);
    }

    /**
     * @brief Returns the transposed version of the tensor by reversing all dimensions.
     * @return The transposed tensor metadata.
     */
    TensorMeta T() {
        std::vector<int> perm = arange(0, ndim());
        std::reverse(perm.begin(), perm.end());

        return permute(perm);
    }

#pragma endregion
};
}  // namespace rash