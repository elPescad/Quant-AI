#include <iostream>
#include <torch/torch.h>

int main() {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << "LibTorch initialized successfully:\n" << tensor << std::endl;
    return 0;
}