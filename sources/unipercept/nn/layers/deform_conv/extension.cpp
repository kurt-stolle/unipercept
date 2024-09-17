#include "extension.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Deformable Convolution (DeformConv) PyTorch extensions.";
  m.def("deform_conv_forward", &deform_conv_forward, "deform_conv_forward");
  m.def("deform_conv_backward", &deform_conv_backward, "deform_conv_backward");
}
