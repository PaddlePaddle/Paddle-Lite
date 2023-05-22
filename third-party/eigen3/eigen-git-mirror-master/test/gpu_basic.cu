// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2015-2016 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// workaround issue between gcc >= 4.7 and cuda 5.5
#if (defined __GNUC__) && (__GNUC__>4 || __GNUC_MINOR__>=7)
  #undef _GLIBCXX_ATOMIC_BUILTINS
  #undef _GLIBCXX_USE_INT128
#endif

#define EIGEN_TEST_NO_LONGDOUBLE
#define EIGEN_TEST_NO_COMPLEX
#define EIGEN_DEFAULT_DENSE_INDEX_TYPE int

#include "main.h"
#include "gpu_common.h"

// Check that dense modules can be properly parsed by nvcc
#include <Eigen/Dense>

// struct Foo{
//   EIGEN_DEVICE_FUNC
//   void operator()(int i, const float* mats, float* vecs) const {
//     using namespace Eigen;
//   //   Matrix3f M(data);
//   //   Vector3f x(data+9);
//   //   Map<Vector3f>(data+9) = M.inverse() * x;
//     Matrix3f M(mats+i/16);
//     Vector3f x(vecs+i*3);
//   //   using std::min;
//   //   using std::sqrt;
//     Map<Vector3f>(vecs+i*3) << x.minCoeff(), 1, 2;// / x.dot(x);//(M.inverse() *  x) / x.x();
//     //x = x*2 + x.y() * x + x * x.maxCoeff() - x / x.sum();
//   }
// };

template<typename T>
struct coeff_wise {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    T x1(in+i);
    T x2(in+i+1);
    T x3(in+i+2);
    Map<T> res(out+i*T::MaxSizeAtCompileTime);
    
    res.array() += (in[0] * x1 + x2).array() * x3.array();
  }
};

template<typename T>
struct replicate {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    T x1(in+i);
    int step   = x1.size() * 4;
    int stride = 3 * step;
    
    typedef Map<Array<typename T::Scalar,Dynamic,Dynamic> > MapType;
    MapType(out+i*stride+0*step, x1.rows()*2, x1.cols()*2) = x1.replicate(2,2);
    MapType(out+i*stride+1*step, x1.rows()*3, x1.cols()) = in[i] * x1.colwise().replicate(3);
    MapType(out+i*stride+2*step, x1.rows(), x1.cols()*3) = in[i] * x1.rowwise().replicate(3);
  }
};

template<typename T>
struct redux {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    int N = 10;
    T x1(in+i);
    out[i*N+0] = x1.minCoeff();
    out[i*N+1] = x1.maxCoeff();
    out[i*N+2] = x1.sum();
    out[i*N+3] = x1.prod();
    out[i*N+4] = x1.matrix().squaredNorm();
    out[i*N+5] = x1.matrix().norm();
    out[i*N+6] = x1.colwise().sum().maxCoeff();
    out[i*N+7] = x1.rowwise().maxCoeff().sum();
    out[i*N+8] = x1.matrix().colwise().squaredNorm().sum();
  }
};

template<typename T1, typename T2>
struct prod_test {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T1::Scalar, T1::RowsAtCompileTime, T2::ColsAtCompileTime> T3;
    T1 x1(in+i);
    T2 x2(in+i+1);
    Map<T3> res(out+i*T3::MaxSizeAtCompileTime);
    res += in[i] * x1 * x2;
  }
};

template<typename T1, typename T2>
struct diagonal {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T1::Scalar* in, typename T1::Scalar* out) const
  {
    using namespace Eigen;
    T1 x1(in+i);
    Map<T2> res(out+i*T2::MaxSizeAtCompileTime);
    res += x1.diagonal();
  }
};

template<typename T>
struct eigenvalues_direct {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T::Scalar, T::RowsAtCompileTime, 1> Vec;
    T M(in+i);
    Map<Vec> res(out+i*Vec::MaxSizeAtCompileTime);
    T A = M*M.adjoint();
    SelfAdjointEigenSolver<T> eig;
    eig.computeDirect(A);
    res = eig.eigenvalues();
  }
};

template<typename T>
struct eigenvalues {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    typedef Matrix<typename T::Scalar, T::RowsAtCompileTime, 1> Vec;
    T M(in+i);
    Map<Vec> res(out+i*Vec::MaxSizeAtCompileTime);
    T A = M*M.adjoint();
    SelfAdjointEigenSolver<T> eig;
    eig.compute(A);
    res = eig.eigenvalues();
  }
};

template<typename T>
struct matrix_inverse {
  EIGEN_DEVICE_FUNC
  void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
  {
    using namespace Eigen;
    T M(in+i);
    Map<T> res(out+i*T::MaxSizeAtCompileTime);
    res = M.inverse();
  }
};

EIGEN_DECLARE_TEST(gpu_basic)
{
  ei_test_init_gpu();
  
  int nthreads = 100;
  Eigen::VectorXf in, out;
  
  #if !defined(__CUDA_ARCH__) && !defined(__HIP_DEVICE_COMPILE__)
  int data_size = nthreads * 512;
  in.setRandom(data_size);
  out.setRandom(data_size);
  #endif
  
  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(coeff_wise<Array44f>(), nthreads, in, out) );

#if !defined(EIGEN_USE_HIP)
  // FIXME
  // These subtests result in a compile failure on the HIP platform
  //
  //  eigen-upstream/Eigen/src/Core/Replicate.h:61:65: error:
  //           base class 'internal::dense_xpr_base<Replicate<Array<float, 4, 1, 0, 4, 1>, -1, -1> >::type'
  //           (aka 'ArrayBase<Eigen::Replicate<Eigen::Array<float, 4, 1, 0, 4, 1>, -1, -1> >') has protected default constructor
  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array4f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(replicate<Array33f>(), nthreads, in, out) );
#endif
  
  CALL_SUBTEST( run_and_compare_to_gpu(redux<Array4f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(redux<Matrix3f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix3f,Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(prod_test<Matrix4f,Vector4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix3f,Vector3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(diagonal<Matrix4f,Vector4f>(), nthreads, in, out) );

  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix2f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(matrix_inverse<Matrix4f>(), nthreads, in, out) );
  
  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix3f>(), nthreads, in, out) );
  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues_direct<Matrix2f>(), nthreads, in, out) );

#if defined(__NVCC__)
  // FIXME
  // These subtests compiles only with nvcc and fail with HIPCC and clang-cuda
  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix4f>(), nthreads, in, out) );
  typedef Matrix<float,6,6> Matrix6f;
  CALL_SUBTEST( run_and_compare_to_gpu(eigenvalues<Matrix6f>(), nthreads, in, out) );
#endif
}
