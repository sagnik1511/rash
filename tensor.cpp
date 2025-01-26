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
    std::vector<std::shared_ptr<Tensor>> prev;

   public:
    static std::vector<std::shared_ptr<Tensor>> allTensors;
    Tensor(double data, bool requiresGrad = false, std::string tag = "") : data_(data), requiresGrad(requiresGrad) {
        if (tag == "")
            this->tag = "tensor_" + std::to_string(++tensorCounter);
        else
            this->tag = tag;

        allTensors.push_back(std::make_shared<Tensor>(*this));
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
        out._backward = [thisData = this->data_, otherData = other.data_, thisGrad = this->requiresGrad,
                         otherGrad = other.requiresGrad, prevThis = std::make_shared<Tensor>(*this),
                         prevOther = std::make_shared<Tensor>(other)](double incGrad) {
            std::cout << "Running +Grad on " << prevThis->tag << " and " << prevOther->tag << "\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";
            if (thisGrad) {
                prevThis->grad += 1.0 * incGrad;
                std::cout << "Grad Value of " << prevThis->tag << " now : " << prevThis->grad << std::endl;
            }

            if (otherGrad) {
                prevOther->grad += 1.0 * incGrad;
                std::cout << "Grad Value of " << prevOther->tag << " now : " << prevOther->grad << std::endl;
            }
        };
        out.prev = {std::make_shared<Tensor>(*this), std::make_shared<Tensor>(other)};

        return out;
    }

    Tensor operator-() {
        Tensor out = Tensor(-data_, requiresGrad, "(-" + tag + ")");
        out._backward = [thisData = this->data_, thisGrad = this->requiresGrad,
                         prevThis = std::make_shared<Tensor>(*this)](double incGrad) {
            std::cout << "Running -Grad on " << prevThis->tag << "\n";
            std::cout << "Incoming Gradient : " << incGrad << "\n";
            if (thisGrad) {
                std::cout << "Grad Value of " << prevThis->tag << " now : " << prevThis->grad << std::endl;
                prevThis->grad += -1 * incGrad;
            }
        };
        out.prev = {std::make_shared<Tensor>(*this)};

        return out;
    }

    Tensor operator-(Tensor& other) {
        Tensor neg = -other;
        return *this + neg;
    }

    void backward(bool isRoot = true) {
        if (isRoot)
            grad = 1.0;

        if (requiresGrad && _backward) {
            std::cout << "Running backward of " << tag << "\n";
            _backward(grad);

            for (auto& tensor : prev) {
                tensor->backward(false);
            }
        }
    }
};

int Tensor::tensorCounter = 0;
std::vector<std::shared_ptr<Tensor>> Tensor::allTensors;

int main() {
    Tensor a = Tensor(2, true, "a");
    Tensor b = Tensor(10, true, "b");
    Tensor c = Tensor(11, true, "c");

    Tensor d = a + b;
    Tensor e = c - d;

    std::cout << "Before Backward Pass" << "\n" << c << std::endl;
    for (const auto& tensor : Tensor::allTensors) {
        std::cout << *tensor << "\n";
    }

    e.backward();

    std::cout << "After Backward Pass" << "\n" << c << std::endl;
    for (const auto& tensor : Tensor::allTensors) {
        std::cout << *tensor << "\n";
    }

    return 0;
}