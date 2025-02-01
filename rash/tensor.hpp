#include <math.h>

#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <vector>

const char* bool2String(bool val) { return val ? "true" : "false"; }

class Tensor {
    double data_, grad = 0.0f;
    std::string tag;
    bool requiresGrad;
    static int tensorCounter;
    std::function<void(double)> _backward;
    std::vector<Tensor*> prev;
    std::map<std::string, bool> gradVisited;

   public:
    static std::vector<Tensor*> allTensors;
    Tensor(double data, bool requiresGrad = false, std::string tensorTag = "")
        : data_(data), requiresGrad(requiresGrad) {
        if (tensorTag == "")
            this->tag = "tensor_" + std::to_string(++tensorCounter);
        else
            this->tag = tensorTag;

        allTensors.push_back(this);
    }

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
        out._backward = [this, &other](double incGrad) {
            std::cout << "Running +Grad on '" << this->tag << "' and '" << other.tag << "'\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";
            if (this->requiresGrad) {
                this->grad += 1.0 * incGrad;
                std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad += 1.0 * incGrad;
                std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};

        return out;
    }

    Tensor operator-() {
        std::string newTag = "(-" + tag + ")";
        Tensor out = Tensor(-data_, requiresGrad, newTag);
        out._backward = [this](double incGrad) {
            std::cout << "Running -Grad on '" << this->tag << "'\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";
            if (this->requiresGrad) {
                this->grad += -1.0 * incGrad;
                std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
        };
        out.prev = {this};

        return out;
    }

    Tensor operator-(Tensor& other) {
        std::string newTag = "(" + tag + "-" + other.tag + ")";
        Tensor out = Tensor(data_ - other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](double incGrad) {
            std::cout << "Running -Grad on '" << this->tag << "' and '" << other.tag << "'\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";
            if (this->requiresGrad) {
                this->grad += 1.0 * incGrad;
                std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }

            if (other.requiresGrad) {
                other.grad += -1.0 * incGrad;
                std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};

        return out;
    }

    Tensor operator*(Tensor& other) {
        std::string newTag = "(" + tag + "*" + other.tag + ")";
        Tensor out = Tensor(data_ * other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](double incGrad) {
            std::cout << "Running *Grad on '" << this->tag << "' and '" << other.tag << "'\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";

            if (this->requiresGrad) {
                this->grad += other.data_ * incGrad;
                std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad += this->data_ * incGrad;
                std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};
        return out;
    }

    Tensor operator/(Tensor& other) {
        std::string newTag = "(" + tag + "/" + other.tag + ")";
        Tensor out = Tensor(data_ / other.data_, requiresGrad || other.requiresGrad, newTag);
        out._backward = [this, &other](double incGrad) {
            std::cout << "Running /Grad on '" << this->tag << "' and '" << other.tag << "'\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";

            if (this->requiresGrad) {
                this->grad += (1 / other.data_) * incGrad;
                std::cout << "Grad Value of " << this->tag << " now : " << this->grad << std::endl;
            }
            if (other.requiresGrad) {
                other.grad += -(this->data_ / (other.data_ * other.data_)) * incGrad;
                std::cout << "Grad Value of " << other.tag << " now : " << other.grad << std::endl;
            }
        };
        out.prev = {this, &other};
        return out;
    }

    void backward(bool isRoot = true) {
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
            std::cout << "Running backward of " << tag << "\n";
            _backward(grad);

            if (prev.size() == 0) {
                std::cout << this->tag << " is leaf node!" << std::endl;
            }
            for (auto& tensor : prev) {
                tensor->backward(false);
            }
        }
    }
};