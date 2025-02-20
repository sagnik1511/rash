#include <cblas.h>

#include <algorithm>
#include <cassert>
#include <functional>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <vector>

// This class is to hold all tensor data
// and it's different orientations/views
class TensorMeta {
    int numel;
    std::vector<int> tensorSize;

   public:
    std::vector<double> rawData;
    TensorMeta(std::vector<double> data, std::vector<int> size) : tensorSize(size), rawData(data) {
        numel = 1;
        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        if (rawData.size() != numel) {
            throw std::runtime_error("Data size mismatch with tensorSize!");
        }
    }
    TensorMeta(double data) : tensorSize({1}), rawData({data}) { numel = 1; }

    TensorMeta(std::vector<int> size) : tensorSize(size) {
        numel = 1;

        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        rawData.assign(numel, 0.0);
        fillRandomData();
    }
    TensorMeta() = default;
    ~TensorMeta() = default;
    TensorMeta(const TensorMeta& other) : numel(other.numel), tensorSize(other.tensorSize), rawData(other.rawData) {}

    void fillRandomData() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < numel; i++) {
            rawData[i] = dis(gen);
        }
    }

    void updateAll(double value) { rawData.assign(numel, value); }
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

    static void display(std::ostream& oss, const TensorMeta& meta) {
        showRecursive(oss, meta.tensorSize, meta.rawData, 0);
    }

    friend std::ostream& operator<<(std::ostream& os, const TensorMeta& meta) {
        meta.display(os, meta);
        return os;
    }

    TensorMeta squeeze(std::vector<int> dims) const {
        std::vector<int> newSize(tensorSize);
        std::sort(dims.begin(), dims.end(), std::greater<>());
        for (auto& idx : dims) {
            if (idx < ndim() && tensorSize[idx] == 1)
                newSize.erase(newSize.begin() + idx);
        }

        return TensorMeta(rawData, newSize);
    }

    TensorMeta squeeze(int dim = 0) const {
        std::vector<int> dims = {dim};
        return squeeze(dims);
    }

    TensorMeta unsqueeze(int dim = 0) const {
        std::vector<int> newSize(tensorSize);
        newSize.insert(newSize.begin() + dim, 1);
        return TensorMeta(rawData, newSize);
    }

    TensorMeta permute(std::vector<int> dims) const {
        int n = ndim();
        assert(dims.size() == ndim() && "Dim Size not matching");
        std::vector<int> newDims;
        std::map<int, int> dimUpdate;
        for (auto& dim : dims) {
            if (!dimUpdate[dim] && dim < n) {
                dimUpdate[dim] = tensorSize[dim];
            } else {
                throw std::runtime_error("Failed due to inconsistent dims");
            }
        }
        for (auto& dim : dims) {
            newDims.push_back(tensorSize[dim]);
        }
        return TensorMeta(rawData, newDims);
    }

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

    static std::vector<int> fetchBroadcastedSize(const TensorMeta& dat1, const TensorMeta& dat2) {
        return fetchBroadcastedSize(dat1.tensorSize, dat2.tensorSize);
    }

    static std::vector<int> fetchStride(const TensorMeta& data) { return fetchStride(data.tensorSize); }

    static std::vector<int> fetchStride(const std::vector<int>& shape) {
        int currDimStride = 1;
        std::vector<int> stride(shape.size(), 0);
        for (int idx = shape.size() - 1; idx >= 0; idx--) {
            stride[idx] = currDimStride;
            // std::cout << stride[idx] << std::endl;
            currDimStride *= shape[idx];
        }

        return stride;
    }

    static int getIndex(const std::vector<int>& indices, const std::vector<int>& shape,
                        const std::vector<int>& stride) {
        int idx = 0;
        int dimOffset = shape.size() - indices.size();
        for (size_t i = 0; i < indices.size(); ++i) {
            int effIdx = (shape[i + dimOffset] == 1) ? 0 : indices[i];
            idx += effIdx * stride[i + dimOffset];
        }
        return idx;
    }

    static void iterateBroadcasting(const TensorMeta& dat1, const TensorMeta& dat2, const std::vector<int>& stride1,
                                    const std::vector<int>& stride2, TensorMeta& output, std::vector<int>& indices,
                                    int dim, const std::function<double(double, double)>& operation) {
        if (dim == output.tensorSize.size()) {
            int idx1, idx2, finIdx;
            idx1 = getIndex(indices, dat1.tensorSize, stride1);
            idx2 = getIndex(indices, dat2.tensorSize, stride2);
            finIdx = getIndex(indices, output.tensorSize, fetchStride(output));

            output.rawData[finIdx] = operation(dat1.rawData[idx1], dat2.rawData[idx2]);
            return;
        }
        for (int iter = 0; iter < output.tensorSize[dim]; iter++) {
            indices[dim] = iter;
            iterateBroadcasting(dat1, dat2, stride1, stride2, output, indices, dim + 1, operation);
        }
    }

    static TensorMeta broadcast(const TensorMeta& dat1, const TensorMeta& dat2,
                                std::function<double(double, double)> op) {
        std::vector<int> broadcastedShape = fetchBroadcastedSize(dat1, dat2);
        std::vector<int> stride1, stride2, indices;
        stride1 = fetchStride(dat1);
        stride2 = fetchStride(dat2);
        indices.assign(broadcastedShape.size(), 0);
        TensorMeta broadcastedMetaStorage(broadcastedShape);
        iterateBroadcasting(dat1, dat2, stride1, stride2, broadcastedMetaStorage, indices, 0, op);

        return broadcastedMetaStorage;
    }

    int ndim() const { return tensorSize.size(); }

    void shape() {
        std::cout << "Shape : ";
        for (auto& el : tensorSize) {
            std::cout << el << ", ";
        }
        std::cout << "\n";
    }

    TensorMeta operator+(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator+(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        TensorMeta otherMeta = TensorMeta({other}, {1});
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    TensorMeta& operator+=(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        *this = std::move(TensorMeta::broadcast(*this, other, op));
        return *this;
    }

    TensorMeta operator-() {
        TensorMeta other({0}, {1});
        std::function<double(double, double)> op = [](double val1, double val2) { return val2 - val1; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator-(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator-(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        TensorMeta otherMeta = TensorMeta({other}, {1});
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    TensorMeta& operator-=(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        *this = std::move(TensorMeta::broadcast(*this, other, op));
        return *this;
    }

    // TensorMeta operator*(const TensorMeta& other) {
    //     std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
    //     return TensorMeta::broadcast(*this, other, op);
    // }

    TensorMeta operator*(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator*(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    TensorMeta operator/(const TensorMeta& other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 / val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator/(double other) const {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 / val2; };
        TensorMeta otherMeta = TensorMeta(other);
        return TensorMeta::broadcast(*this, otherMeta, op);
    }

    static TensorMeta exp(const TensorMeta& meta) {
        TensorMeta other(1);
        std::function<double(double, double)> op = [](double val1, double val2) { return std::exp(val1); };
        return TensorMeta::broadcast(meta, other, op);
    }

    static TensorMeta abs(const TensorMeta& meta) {
        TensorMeta other(1);
        std::function<double(double, double)> op = [](double val1, double val2) { return std::abs(val1); };
        return TensorMeta::broadcast(meta, other, op);
    }

    operator double() const {
        if (ndim() == 1 && tensorSize[0] == 1) {
            return rawData[0];
        } else {
            throw std::runtime_error("Higher Dimensional data can't be converted to Scalar-type");
        }
    }

    // static std::vector<int> fetchMatmulSize(const TensorMeta& dat1, const TensorMeta& dat2) {
    //     return fetchMatmulSize(dat1.tensorSize, dat2.tensorSize);
    // }

    // static std::vector<int> fetchMatMulSize(const std::vector<int> sz1, const std::vector<int> sz2) {
    //     assert(sz1.size() >= 2 && sz2.size() >= 2 && "MatMul Operation needs atleast 2 dimensions!");
    // }

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
                std::cout << e.what() << std::endl;
                return false;
            }
        }
    }

    // TODO:Update for broadcasting and all edge cases
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
            // std::cout << "Iteration : " << batchIdx << "\n--------------\n";
            // std::cout << "OffSet1 : " << offSet1 << std::endl;
            // std::cout << "OffSet2 : " << offSet2 << std::endl;
            // std::cout << "OffSetOut : " << offSetOut << std::endl;
            // std::cout << "--------------\n";

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

    std::vector<int> fetchSqueezedShape(const std::vector<int>& origShape, std::vector<int> axis = {},
                                        bool keepdims = false) {
        if (!axis.size()) {
            return {1};
        }
        std::set<int> axes(axis.begin(), axis.end());
        std::vector<int> finShape(origShape.begin(), origShape.end());
        std::set<int>::reverse_iterator it;
        for (it = axes.rbegin(); it != axes.rend(); ++it) {
            int dim = *it;
            if (dim < ndim() && dim >= 0) {
                if (keepdims) {
                    finShape[dim] = 1;
                } else {
                    finShape.erase(finShape.begin() + dim);
                }
            }
        }
        return finShape;
    }

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

        // std::cout << baseIdx << " + " << outIdx << "\n";
        //  In place Sum Operation
        outMeta[outIdx] += baseMeta[baseIdx];
    }

    TensorMeta sum(std::vector<int> axis = {}, bool keepdims = false) {
        TensorMeta out(fetchSqueezedShape(tensorSize, axis, keepdims));
        out.updateAll(0.0);
        std::vector<int> indices(ndim(), 0);
        std::vector<int> stride = fetchStride(tensorSize);
        std::vector<int> outStride = fetchStride(out.tensorSize);

        for (int b = 0; b < numel; ++b) {
            squeezedSumAtomic(indices, rawData, out.rawData, tensorSize, stride, out.tensorSize, outStride, axis,
                              keepdims);
            for (int dim = indices.size() - 1; dim >= 0; --dim) {
                indices[dim]++;
                if (indices[dim] < tensorSize[dim]) {
                    break;
                }
                indices[dim] = 0;
            }
        }

        return out;
    }

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
};