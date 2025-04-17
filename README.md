# 📄 RASH

A lightweight C++ Header only Library for Tensor Manipulations and Neural Networks!  

**RASH** was born from a curiosity to deeply understand the internal workings of tensors, dynamic tensor definitions, and optimization functionalities.
While it may not yet achieve the performance of production-grade libraries, it showcases smart and flexible design choices, enabling tensors to perform a wide range of operations with ease.
This project lays the structural foundation of a complete tensor ecosystem, with the hope of expanding its capabilities over time — because imagination knows no limits!

---

## 📦 Features

- 🚀 Fast and lightweight
- 📚 Well-documented API (via Doxygen)
- 🐳 Easy integration (header-only)
- 🔥 Supports C++17 and later
- By the way, it currently runs on the CPU — but plans are in place to unleash its full power on GPUs soon! 😉

---

## 🏗️ Installation / Usage

### 🔧 Using Locally

Clone the repository and include the header files in your project:

```bash
git clone https://github.com/sagnik1511/rash.git
```

In your C++ project, just add the `rash/` folder to your include path and include the necessary headers:

```cpp
#include "rash/rash.hpp"
```

No compilation or building necessary — it's header-only!

---

## 🐳 Running with Docker

No local setup needed!  
Use the provided `Dockerfile` to build and run a sample project inside a container:

```bash
# Build the Docker image
docker build -t rash .

# Run the container
docker run --rm rash
```

(You can modify the `Dockerfile` to mount your project or adjust configurations.)

---

## 📚 Documentation

The full project documentation is generated with [Doxygen](https://www.doxygen.nl/).

You can view it [**here**](https://sagnik1511.github.io/rash/) (hosted via GitHub Pages).

To generate it manually:

```bash
doxygen Doxyfile
```

The output will be in the `docs/html/` folder.

---

## 📂 Project Structure

```text
.
├── Dockerfile
├── Doxyfile
├── README.md
├── rash/
│   └── (header files)
├── test/
│   └── (test files)
└── docs/
    └── html/ (auto-generated Doxygen documentation)

```

---

## 🛠️ Requirements

- C++17 or later
- Docker (Optional, unless you have ubuntu>22.04 OS present)

---

## ✨ Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Make sure to update tests and documentation as appropriate.

---

## 📄 License

Licensed under the [MIT License](LICENSE).

---

## 🤛‍♂️ Acknowledgements

- Inspired by the best Tensor Lib present - [PyTorch](https://github.com/pytorch/pytorch)
- Documentation powered by Doxygen
