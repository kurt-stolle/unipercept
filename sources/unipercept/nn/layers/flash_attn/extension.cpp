#include "extension.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Multi-Scale Deformable Flash Attention (FlashAttn) PyTorch extensions.";
  m.def("flash_attn_forward", &flash_attn_forward, "flash_attn_forward");
  m.def("flash_attn_backward", &flash_attn_backward, "flash_attn_backward");
}
