#include <math.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

const char* bool2String(bool val) { return val ? "true" : "false"; }

/*
This a fairly simple n-dimensional Tensor Class
which is designed to learn how primitive tensor operations like forward and backward work.
This class also have autograd functionality like PyTorch (unlike it's efficient performance)
Oh, btw for now the n in n-dimensional is 1 :)
*/
class Tensor {
    double data_, grad = 0.0f;
    std::string tag;
    bool requiresGrad, trainable;
    static int tensorCounter;
    std::function<void(double, bool)> _backward;
    std::vector<Tensor*> prev;
    std::map<std::string, bool> gradVisited;

   public:
    static std::vector<Tensor*> allTensors;
    static std::map<std::string, Tensor*> tensors;
    Tensor(double data, bool requiresGrad = false, std::string tensorTag = "", bool trainable = false)
        : data_(data), requiresGrad(requiresGrad), trainable(trainable) {
        if (tensorTag == "")
            this->tag = "tensor_" + std::to_string(++tensorCounter);
        else
            this->tag = tensorTag;

        allTensors.push_back(this);
        if (tag != "")
            tensors[tag] = this;
    }

    Tensor() : data_(0), requiresGrad(false), tag("") {};

    friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor) {
        os << "Tensor(";
        os << tensor.data_ << ", requires_grad=" << bool2String(tensor.requiresGrad) << ", ";
        if (tensor.requiresGrad)
            os << "Grad=" << tensor.grad << ", ";
        os << "Tag=" << tensor.tag;
        os << ")  ";
        return os;
    }

    Tensor operator+(Tensor& other) {
        std::string newTag = "(" + tag + "+" + other.tag + ")";
        Tensor out = Tensor(data_ + other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](double incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running +Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += 1.0 * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad += 1.0 * incGrad;
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
        out._backward = [this](double incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running -Grad on '" << this->tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += -1.0 * incGrad;
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
        out._backward = [this, &other](double incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running -Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += 1.0 * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad += -1.0 * incGrad;
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
        out._backward = [this, &other](double incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running *Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += other.data_ * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad += this->data_ * incGrad;
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
        out._backward = [this, &other](double incGrad, bool verbose = false) {
            if (verbose) {
                std::cout << "Running /Grad on '" << this->tag << "' and '" << other.tag << "'\n";
                std::cout << "Incoming Gradient : " << incGrad << "\n";
            }

            if (this->requiresGrad) {
                this->grad += (1 / other.data_) * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad += -(this->data_ / (other.data_ * other.data_)) * incGrad;
                if (verbose)
                    std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};
        return out;
    }

    Tensor exp() {
        std::string newTag = "exp(" + tag + ")";
        double expVal = std::__math::exp(this->data_);
        Tensor out = Tensor(expVal, requiresGrad, newTag);
        out._backward = [this, expVal](double incGrad, bool verbose = false) {
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
        // Skip if already visited
        if (gradVisited[this->tag]) {
            return;
        }
        gradVisited[this->tag] = true;

        if (isRoot) {
            grad = 1.0;
            gradVisited.clear();
        }

        if (requiresGrad && _backward) {
            if (verbose)
                std::cout << "Running backward of " << tag << "\n";
            _backward(grad, verbose);

            if (prev.size() == 0) {
                if (verbose)
                    std::cout << this->tag << " is leaf node!" << std::endl;
            }
            for (auto& tensor : prev) {
                tensor->backward(false);
            }
        }
    }

    void removeNonTrainableParamsFromCompGraph() {
        for (auto tensor : Tensor::allTensors) {
            std::cout << *tensor << std::endl;
        }
    }

    double fetchData() { return data_; }
    double fetchGrad() { return grad; }
    void updateData(double value) { this->data_ = value; }
    void updaetGrad(double value) { this->grad = value; }
};