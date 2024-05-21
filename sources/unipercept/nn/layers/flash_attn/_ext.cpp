#include "_ext.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("flash_attn_forward", &flash_attn_forward, "flash_attn_forward");
  m.def("flash_attn_backward", &flash_attn_backward, "flash_attn_backward");
}
