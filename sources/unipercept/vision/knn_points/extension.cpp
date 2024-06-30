#include <torch/extension.h>
#include "extension.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PointCloud K-nearest neighbours (P-KNN) PyTorch extensions.";
  m.def("knn_points_idx", &KNearestNeighborIdx);
  m.def("knn_points_backward", &KNearestNeighborBackward);
#ifdef WITH_CUDA
  m.def("knn_check_version", &KnnCheckVersion);
#endif
}
