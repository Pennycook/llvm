#include <cstdio>
#include <cstdlib>

#include <sycl/sycl.hpp>

namespace syclex = sycl::ext::oneapi::experimental;

float sum(sycl::queue q, float* input, size_t N) {

  float result = 0;
  {
    sycl::buffer<float> buf{&result, 1};

    q.submit([&](sycl::handler& h) {
      auto reduction = sycl::reduction(buf, h, sycl::plus<>(), syclex::properties(syclex::deterministic));
      h.parallel_for(N, reduction, [=](size_t i, auto& reducer) {
        reducer += input[i];
      });
    });
  }
  return result;

}

int main(int argc, char* argv[]) {

  constexpr size_t N = 1024;
  float* array = new float[N];

  sycl::queue q;
  float x = sum(q, array, N);
  float y = sum(q, array, N);

  // NB: determinism guarantees bitwise reproducible reductions for floats
  assert(sycl::bit_cast<unsigned int>(x) == sycl::bit_cast<unsigned int>(y));

  delete[] array;
}
