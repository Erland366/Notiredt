#include <torch/torch.h>

int main()
{
    torch::Tensor tensor = torch::rand({6, 6});
    std::cout << tensor << std::endl;
    return 0;
}