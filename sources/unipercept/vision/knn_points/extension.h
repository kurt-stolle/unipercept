#pragma once
#include <torch/extension.h>
#include <tuple>



#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor.")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x " must be contiguous.")
#define CHECK_CONTIGUOUS_CUDA(x) \
  CHECK_CUDA(x);                 \
  CHECK_CONTIGUOUS(x)

// CPU implementation.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K);

// CUDA implementation
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdxCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K,
    const int version);

// Implementation which is exposed.
std::tuple<at::Tensor, at::Tensor> KNearestNeighborIdx(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const int norm,
    const int K,
    const int version) {
  if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(p1);
    CHECK_CUDA(p2);
    return KNearestNeighborIdxCuda(
        p1, p2, lengths1, lengths2, norm, K, version);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return KNearestNeighborIdxCpu(p1, p2, lengths1, lengths2, norm, K);
}

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCpu(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists);

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackwardCuda(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists);

std::tuple<at::Tensor, at::Tensor> KNearestNeighborBackward(
    const at::Tensor& p1,
    const at::Tensor& p2,
    const at::Tensor& lengths1,
    const at::Tensor& lengths2,
    const at::Tensor& idxs,
    const int norm,
    const at::Tensor& grad_dists) {
  if (p1.is_cuda() || p2.is_cuda()) {
#ifdef WITH_CUDA
    CHECK_CUDA(p1);
    CHECK_CUDA(p2);
    return KNearestNeighborBackwardCuda(
        p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
#else
    AT_ERROR("Not compiled with GPU support.");
#endif
  }
  return KNearestNeighborBackwardCpu(
      p1, p2, lengths1, lengths2, idxs, norm, grad_dists);
}

// Utility to check whether a KNN version can be used (CUDA)
bool KnnCheckVersion(int version, const int64_t D, const int64_t K);
