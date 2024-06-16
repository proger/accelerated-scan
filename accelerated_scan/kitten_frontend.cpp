#include <torch/extension.h>
#include <vector>


extern void  attend(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor f, torch::Tensor o);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "Test handler for warp test"; // optional module docstring
    m.def("attend", attend);
}