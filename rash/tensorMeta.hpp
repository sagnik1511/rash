#include <cblas.h>

#include <cassert>
#include <functional>
#include <iostream>
#include <random>
#include <vector>

// This class is to hold all tensor data
// and it's different orientations/views
class TensorMeta {
    int numel;
    std::random_device rd;
    std::vector<int> tensorSize;
    std::vector<double> rawData;

   public:
    TensorMeta(std::vector<double> data, std::vector<int> size) : tensorSize(size), rawData(data) {
        numel = 1;
        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        if (rawData.size() != numel) {
            throw std::runtime_error("Data size mismatch with tensorSize!");
        }
    }

    TensorMeta(std::vector<int> size) : tensorSize(size) {
        numel = 1;

        for (auto& dim : tensorSize) {
            numel *= dim;
        }
        rawData.assign(numel, 0.0);
        fillRandomData();
    }

    ~TensorMeta() = default;

    // TensorMeta() : tensorSize({0}) {};
    TensorMeta(const TensorMeta& other) : numel(other.numel), tensorSize(other.tensorSize), rawData(other.rawData) {}

    void fillRandomData() {
        std::mt19937 gen;
        std::uniform_real_distribution<> dis(0, 1);

        for (int i = 0; i < numel; i++) {
            rawData[i] = dis(gen);
        }
    }

    void updateAll(double value) { rawData.assign(numel, value); }
    void showRecursive(std::ostream& os, std::vector<int> shape, int startIdx = 0) const {
        if (shape.size() == 1) {
            os << "[";
            for (int i = startIdx; i < startIdx + shape[0] - 1; i++) {
                os << rawData[i] << " ,";
            }
            os << rawData[startIdx + shape[0] - 1] << "]";
        } else {
            std::vector<int> childShape(shape.begin() + 1, shape.end());
            int totalElPerDim = 1;
            for (auto el : childShape) {
                totalElPerDim *= el;
            }
            os << "[";
            for (int i = 0; i < shape[0] - 1; i++) {
                showRecursive(os, childShape, startIdx + (i * totalElPerDim));
                os << ", \n";
            }
            showRecursive(os, childShape, startIdx + ((shape[0] - 1) * totalElPerDim));
            os << "]";
        }
    }

    void display() {
        showRecursive(std::cout, tensorSize, 0);
        std::cout << std::endl;
    }

    static std::vector<int> fetchBroadcastedSize(const TensorMeta& dat1, const TensorMeta& dat2) {
        if (!dat1.tensorSize.size() || !dat2.tensorSize.size()) {
            throw std::length_error("Tensor should have atleats a dimension!");
            std::exit(-1);
        }

        std::vector<int> curr(dat1.tensorSize.rbegin(), dat1.tensorSize.rend()),
            incm(dat2.tensorSize.rbegin(), dat2.tensorSize.rend());

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
                    std::exit(-1);
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

    static std::vector<int> fetchStride(const TensorMeta& data) {
        int currDimStride = 1;
        std::vector<int> stride(data.tensorSize.size(), 0);
        for (int idx = data.tensorSize.size() - 1; idx >= 0; idx--) {
            stride[idx] = currDimStride;
            // std::cout << stride[idx] << std::endl;
            currDimStride *= data.tensorSize[idx];
        }

        return stride;
    }

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

    void shape() {
        std::cout << "Shape : ";
        for (auto& el : tensorSize) {
            std::cout << el << ", ";
        }
        std::cout << "\n";
    }

    TensorMeta operator+(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 + val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator-(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 - val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator*(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 * val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    TensorMeta operator/(const TensorMeta& other) {
        std::function<double(double, double)> op = [](double val1, double val2) { return val1 / val2; };
        return TensorMeta::broadcast(*this, other, op);
    }

    static bool isAbletoPerformMatmul(const TensorMeta& dat1, const TensorMeta& dat2) {
        // The process doesn't allow broadcasting in matmul as of now

        std::vector<int> v1, v2;
        v1 = dat1.tensorSize;
        v2 = dat2.tensorSize;

        std::reverse(v1.begin(), v1.end());
        std::reverse(v2.begin(), v2.end());

        if (v1.size() < 2 || v1.size() < 2) {
            std::cerr << "Not enough dimensions to perform matmul operation\n";
            return false;
        }
        if (v1[0] != v2[1]) {
            std::cerr << "Dimension mismatch for matmul operation\n";
            return false;
        }
        for (int idx = 2; idx < v1.size(); idx++) {
            if (v1[idx] != v2[idx]) {
                std::cerr << "Dimension mismatch, possibly as broadcasting not allowed\n";
                return false;
            }
        }
        return true;
    }

    static std::vector<int> fetchMatmulShape(const TensorMeta& dat1, const TensorMeta& dat2) {
        bool matMulexecFlag = isAbletoPerformMatmul(dat1, dat2);
        if (!matMulexecFlag) {
            throw std::runtime_error("");
        }
        std::vector<int> matmulSize = dat1.tensorSize;
        matmulSize[matmulSize.size() - 1] = dat2.tensorSize.back();

        return matmulSize;
    }

    static void matmulAtomic(const std::vector<double>& A, const std::vector<double>& B, std::vector<double>& out,
                             int offSetA, int offSetB, int offSetOut, int M, int N, int K) {
        // std::cout << "M : "<< M << std::endl;
        // std::cout << "N : "<< N << std::endl;
        // std::cout << "K : "<< K << std::endl;

        // std::cout << "A Size : "<< A.size() << std::endl;
        // std::cout << "B Size : "<< B.size() << std::endl;
        // std::cout << "out Size : "<< out.size() << std::endl;

        assert(A.size() >= offSetA + M * N && "A vector is too small!");
        assert(B.size() >= offSetB + N * K && "B vector is too small!");
        assert(out.size() >= offSetOut + M * K && "Output vector is too small!");
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, K, N, 1.0, &A[offSetA], N, &B[offSetB], K, 0.0,
                    &out[offSetOut], K);
    }

    static TensorMeta matmul(const TensorMeta& dat1, const TensorMeta& dat2) {
        TensorMeta out(fetchMatmulShape(dat1, dat2));
        int M = dat1.tensorSize[dat1.tensorSize.size() - 2];
        int N = dat1.tensorSize[dat1.tensorSize.size() - 1];
        int P = dat2.tensorSize[dat2.tensorSize.size() - 1];
        matmulAtomic(dat1.rawData, dat2.rawData, out.rawData, 0, 0, 0, M, N, P);

        return out;
    }
};