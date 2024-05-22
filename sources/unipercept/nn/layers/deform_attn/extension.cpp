#include "extension.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Multi-Scale Deformable Attention (MultiScaleDeformAttn) PyTorch extensions.";
  m.def("deform_attn_forward", &deform_attn_forward, "deform_attn_forward");
  m.def("deform_attn_backward", &deform_attn_backward, "deform_attn_backward");
}
