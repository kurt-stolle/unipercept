namespace {
template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    typename... Args>
struct DispatchKernelHelper1D {
  static void run(const int64_t N, Args... args) {
    if (curN == N) {
      Kernel<T, curN>::run(args...);
    } else if (curN < N) {
      DispatchKernelHelper1D<Kernel, T, minN, maxN, curN + 1, Args...>::run(
          N, args...);
    }
  }
};
template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    typename... Args>
struct DispatchKernelHelper1D<Kernel, T, minN, maxN, maxN, Args...> {
  static void run(const int64_t N, Args... args) {
    if (N == maxN) {
      Kernel<T, maxN>::run(args...);
    }
  }
};
template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    int64_t minM,
    int64_t maxM,
    int64_t curM,
    typename... Args>
struct DispatchKernelHelper2D {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (curN == N && curM == M) {
      Kernel<T, curN, curM>::run(args...);
    } else if (curN < N && curM < M) {
      // Increment both curN and curM. This isn't strictly necessary; we could
      // just increment one or the other at each step. But this helps to cut
      // on the number of recursive calls we make.
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    } else if (curN < N) {
      // Increment curN only
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          curM,
          Args...>::run(N, M, args...);
    } else if (curM < M) {
      // Increment curM only
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    }
  }
};

template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    int64_t curM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    maxN,
    minM,
    maxM,
    curM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (maxN == N && curM == M) {
      Kernel<T, maxN, curM>::run(args...);
    } else if (curM < maxM) {
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          maxN,
          minM,
          maxM,
          curM + 1,
          Args...>::run(N, M, args...);
    }
  }
};

template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t curN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    curN,
    minM,
    maxM,
    maxM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (curN == N && maxM == M) {
      Kernel<T, curN, maxM>::run(args...);
    } else if (curN < maxN) {
      DispatchKernelHelper2D<
          Kernel,
          T,
          minN,
          maxN,
          curN + 1,
          minM,
          maxM,
          maxM,
          Args...>::run(N, M, args...);
    }
  }
};

template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
struct DispatchKernelHelper2D<
    Kernel,
    T,
    minN,
    maxN,
    maxN,
    minM,
    maxM,
    maxM,
    Args...> {
  static void run(const int64_t N, const int64_t M, Args... args) {
    if (maxN == N && maxM == M) {
      Kernel<T, maxN, maxM>::run(args...);
    }
  }
};

}

template <
    template <typename, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    typename... Args>
void DispatchKernel1D(const int64_t N, Args... args) {
  if (minN <= N && N <= maxN) {
    DispatchKernelHelper1D<Kernel, T, minN, maxN, minN, Args...>::run(
        N, args...);
  }
}

template <
    template <typename, int64_t, int64_t>
    class Kernel,
    typename T,
    int64_t minN,
    int64_t maxN,
    int64_t minM,
    int64_t maxM,
    typename... Args>
void DispatchKernel2D(const int64_t N, const int64_t M, Args... args) {
  if (minN <= N && N <= maxN && minM <= M && M <= maxM) {
    DispatchKernelHelper2D<
        Kernel,
        T,
        minN,
        maxN,
        minN,
        minM,
        maxM,
        minM,
        Args...>::run(N, M, args...);
  }
}
