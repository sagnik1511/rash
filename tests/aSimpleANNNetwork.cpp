// g++ --std=c++17 tests/aSimpleANNNetwork.cpp -o nn -g -lopenblas

#include <math.h>

#include "../rash/rash.hpp"

using namespace rash;

/**
 * @brief Prepares a dataset for training.
 * Data is generated based on a simple non-linear equation Y = F(x) = sin(x).
 * @param numSamples The number of samples to generate.
 * @return A tuple containing the input and output tensors.
 */
std::tuple<TensorMeta, TensorMeta> prepareDataset(int numSamples) {
    std::vector<double> xRaw, yRaw;
    double baseValue = 1.0 / numSamples;

    // Load random indexes
    std::random_device rd;
    std::mt19937 g(rd());
    auto range = arange(0, numSamples);
    std::shuffle(range.begin(), range.end(), g);

    // Fill Raw Data
    for (auto i : range) {
        double xVal = double(i) * baseValue * M_PI * 2.0;
        double yVal = sin(xVal);
        xRaw.push_back(xVal);
        yRaw.push_back(yVal);
    }

    return std::make_tuple(TensorMeta(xRaw, {numSamples, 1}), TensorMeta(yRaw, {numSamples, 1}));
}

int main() {
    // Configuration of the sample ANN
    int numSamples = 100;
    int hiddenDimSize = 15;
    int numIterations = 20000;
    int step = 0;
    double lr = 1e-4;

    // Load Dataset
    auto [Xdat, ydat] = prepareDataset(numSamples);

    // First Layer
    Tensor w1 = Tensor::rand({hiddenDimSize, 1}, true, "W1");
    Tensor b1 = Tensor::rand({hiddenDimSize}, true, "b1");

    // ReLU
    ReLU relu;

    // Hidden Layer
    Tensor w2 = Tensor::rand({1, hiddenDimSize}, true, "W2");
    Tensor b2 = Tensor::rand({1}, true, "b2");

    // Training Loop
    while (step < numIterations) {
        // Make Input Tensor
        Tensor X(Xdat, false, "X");
        Tensor y(ydat, false, "y");

        // Zero Grad
        w1.zeroGrad(), b1.zeroGrad();
        w2.zeroGrad(), b2.zeroGrad();

        // Make Predictions from the ANN
        Tensor hidden = relu(Tensor::matmul(X, w1.T()) + b1);
        hidden.updateTag("Hidden");

        Tensor pred = Tensor::matmul(hidden, w2.T()) + b2;
        pred.updateTag("pred");

        // Loss Function (MSE)
        Tensor diff = (pred - y).pow(2);
        diff.updateTag("MSE");

        std::cout << "Loss at step " << step << " : " << double(diff.fetchData().sum()) << "\n";

        // Run BackPropagation
        diff.backward();

        // Update Grads
        w1.updateData(w1.fetchData() - (w1.fetchGrad() * lr));
        b1.updateData(b1.fetchData() - (b1.fetchGrad() * lr));

        w2.updateData(w2.fetchData() - (w2.fetchGrad() * lr));
        b2.updateData(b2.fetchData() - (b2.fetchGrad() * lr));

        ++step;
    }
    return 0;
}