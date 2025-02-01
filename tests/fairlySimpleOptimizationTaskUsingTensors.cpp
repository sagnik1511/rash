#include <cstdlib>
#include <fstream>
#include <iostream>

#include "../rash/rash.hpp"

int Tensor::tensorCounter = 0;
std::vector<Tensor*> Tensor::allTensors;
std::map<std::string, Tensor*> Tensor::tensors;

// Creates a lineplot of given values and saves into a png file
// Copied from ChatGPT, cause most of my time went resolving dangling tensor pointers
void plotWithGnuplot(const std::vector<double>& values) {
    // Save data to a file
    std::ofstream file("data.dat");
    for (size_t i = 0; i < values.size(); ++i) {
        file << i << " " << values[i] << std::endl;
    }
    file.close();

    // Create a gnuplot script to generate the PNG file
    std::ofstream gnuplotScript("plot.gnu");
    gnuplotScript << "set terminal pngcairo enhanced font 'Arial,12' size 800,600\n";
    gnuplotScript << "set output 'plot.png'\n";
    gnuplotScript << "set title 'Line Chart'\n";
    gnuplotScript << "set xlabel 'Index'\n";
    gnuplotScript << "set ylabel 'Value'\n";
    gnuplotScript << "set grid\n";
    gnuplotScript
        << "plot 'data.dat' with linespoints linestyle 1 linecolor 'blue' pointtype 7 pointsize 1.5 linewidth 2\n";
    gnuplotScript.close();

    // Run the script with gnuplot
    system("gnuplot plot.gnu");
}

// Main Optimization Task
// The veryy huge computational Operation can be written as below -
// F(x) = e^(a+b)
// We're optimizing a and b's weight so that target F(x) value optimizes implicitly
int main() {
    Tensor a = Tensor(5, true, "a");
    Tensor b = Tensor(1, true, "b");

    std::vector<double> lossTracker;

    int iter = 0;
    while (iter < 1000) {
        // Zero-Grad
        for (auto& [tag, tensor] : Tensor::tensors) {
            tensor->updaetGrad(0);
        }

        // Forward Pass
        Tensor c = a + b;
        Tensor d = c.exp();

        // Backward Pass
        d.backward();

        // Weight Update using Grads
        a.updateData(a.fetchData() - a.fetchGrad() * 0.0001);
        b.updateData(b.fetchData() - b.fetchGrad() * 0.0001);

        std::cout << "Updated tensors's data after iteration " << iter << std::endl;

        for (auto& [tag, tensor] : Tensor::tensors) {
            std::cout << *tensor << std::endl;
        }

        lossTracker.push_back(d.fetchData());

        iter++;
    }

    plotWithGnuplot(lossTracker);
}