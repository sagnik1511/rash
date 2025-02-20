#include <math.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "tensorMeta.hpp"

const char* bool2String(bool val) { return val ? "true" : "false"; }

/*
This a fairly simple n-dimensional Tensor Class
which is designed to learn how primitive tensor operations like forward and backward work.
This class also have autograd functionality like PyTorch (unlike it's efficient performance)
*/
class Tensor {
    TensorMeta data_, grad;
    std::string tag;
    bool requiresGrad, trainable;
    static int tensorCounter;
    std::function<void(TensorMeta, bool)> _backward;
    std::vector<Tensor*> prev;
    std::map<std::string, bool> gradVisited;

   public:
    static std::map<std::string, Tensor*> tensors;
    Tensor(TensorMeta data, bool requiresGrad = false, std::string tensorTag = "", bool trainable = false)
        : data_(data), requiresGrad(requiresGrad), trainable(trainable) {
        if (tensorTag == "")
            this->tag = "tensor_" + std::to_string(++tensorCounter);
        else
            this->tag = tensorTag;

        if (tag != "")
            tensors[tag] = this;

        grad = TensorMeta(data);
        grad.updateAll(0.0);
    }
    Tensor(double data, bool requiresGrad = false, std::string tensorTag = "", bool trainable = false)
        : data_(TensorMeta(data)), requiresGrad(requiresGrad), trainable(trainable) {
        if (tensorTag == "")
            this->tag = "tensor_" + std::to_string(++tensorCounter);
        else
            this->tag = tensorTag;

        if (tag != "")
            tensors[tag] = this;

        grad = TensorMeta(data);
        grad.updateAll(0.0);
    }

    Tensor() : data_(TensorMeta()), requiresGrad(false), tag("") {};
    ~Tensor() {
        std::cout << "Destoyed! " << this->tag << std::endl;
        tensors.erase(tag);
    };

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor(";
        os << tensor.data_ << ", requires_grad=" << bool2String(tensor.requiresGrad) << ", ";
        if (tensor.requiresGrad)
            os << "Grad=" << tensor.grad << ", ";
        os << "Tag=" << tensor.tag;
        os << ")  ";
        return os;
    }

    // This function helps the broadcasted tensor meta2 match dimension with meta1
    static TensorMeta squeezeSum(const TensorMeta& dat1, const TensorMeta& dat2) {
        TensorMeta out = dat2;
        auto [addedDims, bcDims] = TensorMeta::fetchBroadcastedAxes(dat1, dat2);
        if (bcDims.size())
            out = out.sum(bcDims);
        if (addedDims.size())
            out = out.sum(addedDims, true);

        return out;
    }

    Tensor operator+(Tensor& other) {
        std::string newTag = "(" + tag + "+" + other.tag + ")";
        Tensor out = Tensor(data_ + other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running +Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += squeezeSum(this->grad, incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad += squeezeSum(other.grad, incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};

        return out;
    }

    Tensor operator-() {
        std::string newTag = "(-" + tag + ")";
        Tensor out = Tensor(-data_, requiresGrad, newTag);
        out._backward = [this](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running -Grad on '" << this->tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad -= squeezeSum(this->grad, incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
        };
        out.prev = {this};

        return out;
    }

    Tensor operator-(Tensor& other) {
        std::string newTag = "(" + tag + "-" + other.tag + ")";
        Tensor out = Tensor(data_ - other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running -Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += squeezeSum(this->grad, incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad -= squeezeSum(other.grad, incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};

        return out;
    }

    Tensor operator*(Tensor& other) {
        std::string newTag = "(" + tag + "*" + other.tag + ")";
        Tensor out = Tensor(data_ * other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running *Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += squeezeSum(this->grad, other.data_ * incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad += squeezeSum(other.grad, this->data_ * incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};
        return out;
    }

    Tensor operator/(Tensor& other) {
        std::string newTag = "(" + tag + "/" + other.tag + ")";
        Tensor out = Tensor(data_ / other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running /Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += squeezeSum(this->grad, incGrad / other.data_);
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad -= squeezeSum(other.grad, (this->data_ / (other.data_ * other.data_)) * incGrad);
                if (verbose)
                    std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};
        return out;
    }

    Tensor exp() {
        std::string newTag = "exp(" + tag + ")";
        TensorMeta expVal = TensorMeta::exp(this->data_);
        Tensor out = Tensor(expVal, requiresGrad, newTag);
        out._backward = [this, expVal](TensorMeta incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running (e^)Grad on '" << this->tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (requiresGrad) {
                this->grad += expVal * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
        };
        out.prev = {this};
        return out;
    }

    void backward(bool isRoot = true, bool verbose = false) {
        if (isRoot) {
            grad.updateAll(1.0);
            gradVisited.clear();
        }

        // Skip if already visited
        if (gradVisited[this->tag]) {
            return;
        }
        gradVisited[this->tag] = true;

        if (requiresGrad && _backward) {
            if (verbose)
                std::cout << "Running backward of " << tag << "\n";
            _backward(grad, verbose);

            if (prev.size() == 0) {
                if (verbose)
                    std::cout << this->tag << " is leaf node!" << std::endl;
            }
            for (auto& tensor : prev) {
                tensor->backward(false, verbose);
            }
        }
    }

    TensorMeta fetchData() { return data_; }
    TensorMeta fetchGrad() { return grad; }
    void updateData(TensorMeta value) { this->data_ = value; }
    void updaetGrad(TensorMeta value) { this->grad = value; }

    void zeroGrad() { this->grad.updateAll(0.0); }
    void updateTag(std::string newTag) { this->tag = newTag; }

    static Tensor rand(const std::vector<int>& shape, bool requiresGrad = false) {
        return Tensor(TensorMeta(shape), requiresGrad);
    }
};