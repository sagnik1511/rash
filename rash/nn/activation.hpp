#pragma once

#include <iostream>

#include "../rash.hpp"

namespace rash {
/**
 * @brief Abstract base class for activation functions.
 */
class Activation {
   public:
    /**
     * @brief Applies the activation function to the input tensor.
     *
     * @param t The input tensor.
     * @return The output tensor after applying the activation function.
     */
    virtual Tensor forward(const Tensor& t) = 0;

    /**
     * @brief General destructor for the Activation class.
     */
    virtual ~Activation() = default;

    /**
     * @brief Applies the activation function to the input tensor.
     *
     * @param t The input tensor.
     * @return The output tensor after applying the activation function.
     */
    Tensor operator()(const Tensor& t) { return forward(t); }
};

/**
 * @brief ReLU (Rectified Linear Unit) activation function.
 *
 * This class implements the ReLU activation function, which is defined as:
 * f(x) = max(0, x)
 */
class ReLU : public Activation {
    /**
     * @brief Applies the ReLU activation function to the input tensor.
     *
     * @param t The input tensor.
     * @return The output tensor after applying the ReLU activation function.
     */
    virtual Tensor forward(const Tensor& t) override {
        std::string newTag = "RELU(" + t.impl->tag + ")";
        TensorMeta mask = t.impl->data_ > 0.0;
        Tensor out(mask * t.impl->data_, t.impl->requiresGrad, newTag);

        out.impl->prev = {t.impl};
        out.impl->_backward = [mask = mask, out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad * mask);
        };
        return out;
    }

   public:
    ReLU() = default;
};

}  // namespace rash