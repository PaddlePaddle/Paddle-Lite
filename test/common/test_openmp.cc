#include <iostream>
#include <omp.h>

int main(void) {
  #pragma omp parallel num_threads(2)
  {
      int thread_id = omp_get_thread_num();
      int nthreads = omp_get_num_threads();
      std::cout << "Hello, OMP " << thread_id << "/" << nthreads << "\n";
  }
  return 0;
}
