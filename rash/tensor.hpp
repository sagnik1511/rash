#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <memory>
#include <vector>

#include "tensorMeta.hpp"

const char* bool2String(bool val) { return val ? "true" : "false"; }

namespace rash {

/**
 * @brief Constructs a TensorImpl object.
 *
 * @param data The tensor's data.
 * @param requiresGrad Flag to indicate if gradients are needed.
 * @param tensorTag A unique identifier for the tensor.
 */
class TensorImpl {
   public:
    std::map<std::string, bool> gradVisited;
    std::function<void(TensorMeta)> _backward;
    std::vector<std::weak_ptr<TensorImpl>> prev;
    bool requiresGrad;
    TensorMeta data_, grad;
    std::string tag;
    TensorImpl(TensorMeta data, bool requiresGrad, std::string tensorTag)
        : data_(std::move(data)), requiresGrad(requiresGrad), tag(tensorTag) {
        grad = TensorMeta(data_.shape());
        grad.updateAll(0.0);
    }

    /**
     * @brief Performs backpropagation through the computation graph.
     */
    void backward() {
        // Return if already calculated the gradients
        if (gradVisited[tag]) {
            return;
        }

        // Mark gradients calculated
        gradVisited[tag] = true;

        // Return if no backward function found!
        if (!(requiresGrad && _backward)) {
            return;
        }

        // Perform gradeint update
        _backward(grad);

        // Backtrack to previous linked tensors
        for (auto& weak_ptr : prev) {
            if (auto p = weak_ptr.lock())
                p->backward();
        }
    }

    /**
     * @brief Resets gradients to zero.
     */
    void zeroGrad() { grad.updateAll(0.0); }

    /**
     * @brief Accumulates gradients, handling broadcasting where necessary.
     *
     * @param incGrad Incoming gradient to be accumulated.
     */
    void accumulateGrad(TensorMeta incGrad) {
        TensorMeta out = incGrad;
        auto [addedDims, bcDims] = TensorMeta::fetchBroadcastedAxes(grad, out);
        if (bcDims.size())
            out = out.sum(bcDims);
        if (addedDims.size())
            out = out.sum(addedDims, true);

        grad += out;
    }

    /**
     * @brief Updates the gradient with a new value.
     *
     * @param updGrad The new gradient value.
     */
    void updateGrad(TensorMeta updGrad) { grad = updGrad; }
};
/**
 * @brief General Tensor Class
 */
class Tensor {
   public:
    std::shared_ptr<TensorImpl> impl;

    /**
     * @brief Constructs a Tensor with specified data.
     */
    Tensor(TensorMeta data, bool requiresGrad, std::string tag)
        : impl(std::make_shared<TensorImpl>(data, requiresGrad, tag)) {}

    /**
     * @brief Constructs a scalar Tensor.
     */
    Tensor(double data, bool requiresGrad, std::string tag)
        : impl(std::make_shared<TensorImpl>(TensorMeta(data), requiresGrad, tag)) {};

    /**
     * @brief Constructs a Tensor from a vector.
     */
    Tensor(std::vector<double> data, std::vector<int> shape, bool requiresGrad, std::string tag)
        : impl(std::make_shared<TensorImpl>(TensorMeta(data, shape), requiresGrad, tag)) {}

    /**
     * @brief Overloads the output stream operator for printing tensors.
     */
    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor(";
        os << tensor.impl->data_ << ", requires_grad=" << bool2String(tensor.impl->requiresGrad) << ", ";
        if (tensor.impl->requiresGrad)
            os << "Grad=" << tensor.impl->grad << ", ";
        os << "Tag=" << tensor.impl->tag;
        os << ")  ";
        return os;
    }

    /**
     * @brief Adds two tensors.
     */
    Tensor operator+(const Tensor& other) {
        std::string newTag = "(" + this->impl->tag + "+" + other.impl->tag + ")";
        Tensor out(this->impl->data_ + other.impl->data_, this->impl->requiresGrad || other.impl->requiresGrad, newTag);

        out.impl->prev = {impl, other.impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();
            auto p1 = out_impl->prev[1].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad);
            if (p1->requiresGrad)
                p1->accumulateGrad(incGrad);
        };

        return out;
    }

    /**
     * @brief Negates a tensor.
     */
    Tensor operator-() {
        std::string newTag = "(-" + impl->tag + ")";
        Tensor out(-impl->data_, impl->requiresGrad, newTag);
        out.impl->prev = {impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(-incGrad);
        };

        return out;
    }

    /**
     * @brief Substracts two tensors.
     */
    Tensor operator-(const Tensor& other) {
        std::string newTag = "(" + this->impl->tag + "-" + other.impl->tag + ")";
        Tensor out(this->impl->data_ - other.impl->data_, this->impl->requiresGrad || other.impl->requiresGrad, newTag);

        out.impl->prev = {impl, other.impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();
            auto p1 = out_impl->prev[1].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad);
            if (p1->requiresGrad)
                p1->accumulateGrad(-incGrad);
        };

        return out;
    }

    /**
     * @brief Multiplies two tensors.
     */
    Tensor operator*(const Tensor& other) {
        std::string newTag = "(" + this->impl->tag + "*" + other.impl->tag + ")";
        Tensor out(this->impl->data_ * other.impl->data_, this->impl->requiresGrad || other.impl->requiresGrad, newTag);

        out.impl->prev = {impl, other.impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();
            auto p1 = out_impl->prev[1].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad * p1->data_);
            if (p1->requiresGrad)
                p1->accumulateGrad(incGrad * p0->data_);
        };

        return out;
    }

    /**
     * @brief Divides two tensors.
     */
    Tensor operator/(const Tensor& other) {
        std::string newTag = "(" + impl->tag + "/" + other.impl->tag + ")";
        Tensor out(impl->data_ / other.impl->data_, impl->requiresGrad || other.impl->requiresGrad, newTag);

        out.impl->prev = {impl, other.impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();
            auto p1 = out_impl->prev[1].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad / p1->data_);
            if (p1->requiresGrad)
                p1->accumulateGrad(-incGrad * (p0->data_ / (p1->data_ * p1->data_)));
        };

        return out;
    }

    /**
     * @brief Greater than comparison of two tensors.
     */
    Tensor operator>(const Tensor& other) {
        std::string newTag = "(" + impl->tag + "/" + other.impl->tag + ")";
        Tensor out(impl->data_ > other.impl->data_, false, newTag);

        return out;
    }

    /**
     * @brief Greater than equals to comparison of two tensors.
     */
    Tensor operator>=(const Tensor& other) {
        std::string newTag = "(" + impl->tag + "/" + other.impl->tag + ")";
        Tensor out(impl->data_ >= other.impl->data_, false, newTag);

        return out;
    }

    /**
     * @brief Less than comparison of two tensors.
     */
    Tensor operator<(const Tensor& other) {
        std::string newTag = "(" + impl->tag + "/" + other.impl->tag + ")";
        Tensor out(impl->data_ < other.impl->data_, false, newTag);

        return out;
    }

    /**
     * @brief Less than equals to comparison of two tensors.
     */
    Tensor operator<=(const Tensor& other) {
        std::string newTag = "(" + impl->tag + "/" + other.impl->tag + ")";
        Tensor out(impl->data_ <= other.impl->data_, false, newTag);

        return out;
    }

    /**
     * @brief Computes element-wise exponential of the tensor.
     */
    Tensor exp() {
        std::string newTag = "exp(" + impl->tag + ")";
        TensorMeta expVal = TensorMeta::exp(impl->data_);
        Tensor out(expVal, impl->requiresGrad, newTag);
        out.impl->prev = {impl};
        out.impl->_backward = [expVal = expVal, out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad * expVal);
        };

        return out;
    }

    /**
     * @brief Returns the transpose of the tensor.
     */
    Tensor T() const {
        std::string newTag = "(" + impl->tag + ").T";
        Tensor out(impl->data_.T(), impl->requiresGrad, newTag);
        out.impl->prev = {impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(incGrad.T());
        };

        return out;
    }

    /**
     * @brief Computes matrix multiplication of two tensors.
     */
    static Tensor matmul(const Tensor& t1, const Tensor& t2) {
        std::string newTag = "(" + t1.impl->tag + "@" + t2.impl->tag + ")";
        Tensor out(TensorMeta::matmul(t1.impl->data_, t2.impl->data_), t1.impl->requiresGrad || t2.impl->requiresGrad,
                   newTag);
        out.impl->prev = {t1.impl, t2.impl};
        out.impl->_backward = [out_impl = out.impl](TensorMeta incGrad) {
            auto p0 = out_impl->prev[0].lock();
            auto p1 = out_impl->prev[1].lock();

            if (p0->requiresGrad)
                p0->accumulateGrad(TensorMeta::matmul(incGrad, p1->data_.transpose()));
            if (p1->requiresGrad)
                p1->accumulateGrad(TensorMeta::matmul(p0->data_.transpose(), incGrad));
        };

        return out;
    }

    /**
     * @brief Resets the gradient to zero.
     */
    void zeroGrad() { impl->zeroGrad(); }

    /**
     * @brief Performs backpropagation from this tensor.
     */
    void backward() {
        impl->gradVisited.clear();
        impl->grad.updateAll(1.0);
        impl->backward();
    }

    /**
     * @brief Generates a random tensor.
     */
    static Tensor rand(const std::vector<int>& shape, bool requiresGrad = false, std::string tensorTag = "") {
        return Tensor(TensorMeta(shape), requiresGrad, tensorTag);
    }

    /**
     * @brief Retrieves the data of the tensor.
     */
    TensorMeta fetchData() const { return impl->data_; }

    /**
     * @brief Retrieves the gradient of the tensor.
     */
    TensorMeta fetchGrad() const { return impl->grad; }

    /**
     * @brief Updates the gradient with a new value.
     */
    void updateGrad(TensorMeta incGrad) {
        assert(incGrad.shape() == impl->grad.shape() && "Grad Shape should match!");
        impl->updateGrad(incGrad);
    }
};

}  // namespace rash