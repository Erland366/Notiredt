#include <torch/extension.h>
#include <vector>

torch::Tensor mean_filter(torch::Tensor image, int radius);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("forward", &mean_filter, "Mean Filter");
}
