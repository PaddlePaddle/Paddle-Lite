// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011 Benoit Jacob <jacob.benoit.1@gmail.com>
// Copyright (C) 2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#define TEST_ENABLE_TEMPORARY_TRACKING
#define EIGEN_NO_STATIC_ASSERT

#include "main.h"

template<typename ArrayType> void vectorwiseop_array(const ArrayType& m)
{
  typedef typename ArrayType::Scalar Scalar;
  typedef Array<Scalar, ArrayType::RowsAtCompileTime, 1> ColVectorType;
  typedef Array<Scalar, 1, ArrayType::ColsAtCompileTime> RowVectorType;

  Index rows = m.rows();
  Index cols = m.cols();
  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  ArrayType m1 = ArrayType::Random(rows, cols),
            m2(rows, cols),
            m3(rows, cols);

  ColVectorType colvec = ColVectorType::Random(rows);
  RowVectorType rowvec = RowVectorType::Random(cols);

  // test addition

  m2 = m1;
  m2.colwise() += colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() + colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) + colvec);

  VERIFY_RAISES_ASSERT(m2.colwise() += colvec.transpose());
  VERIFY_RAISES_ASSERT(m1.colwise() + colvec.transpose());

  m2 = m1;
  m2.rowwise() += rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() + rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) + rowvec);

  VERIFY_RAISES_ASSERT(m2.rowwise() += rowvec.transpose());
  VERIFY_RAISES_ASSERT(m1.rowwise() + rowvec.transpose());

  // test substraction

  m2 = m1;
  m2.colwise() -= colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() - colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) - colvec);

  VERIFY_RAISES_ASSERT(m2.colwise() -= colvec.transpose());
  VERIFY_RAISES_ASSERT(m1.colwise() - colvec.transpose());

  m2 = m1;
  m2.rowwise() -= rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() - rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) - rowvec);

  VERIFY_RAISES_ASSERT(m2.rowwise() -= rowvec.transpose());
  VERIFY_RAISES_ASSERT(m1.rowwise() - rowvec.transpose());

  // test multiplication

  m2 = m1;
  m2.colwise() *= colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() * colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) * colvec);

  VERIFY_RAISES_ASSERT(m2.colwise() *= colvec.transpose());
  VERIFY_RAISES_ASSERT(m1.colwise() * colvec.transpose());

  m2 = m1;
  m2.rowwise() *= rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() * rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) * rowvec);

  VERIFY_RAISES_ASSERT(m2.rowwise() *= rowvec.transpose());
  VERIFY_RAISES_ASSERT(m1.rowwise() * rowvec.transpose());

  // test quotient

  m2 = m1;
  m2.colwise() /= colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() / colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) / colvec);

  VERIFY_RAISES_ASSERT(m2.colwise() /= colvec.transpose());
  VERIFY_RAISES_ASSERT(m1.colwise() / colvec.transpose());

  m2 = m1;
  m2.rowwise() /= rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() / rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) / rowvec);

  VERIFY_RAISES_ASSERT(m2.rowwise() /= rowvec.transpose());
  VERIFY_RAISES_ASSERT(m1.rowwise() / rowvec.transpose());

  m2 = m1;
  // yes, there might be an aliasing issue there but ".rowwise() /="
  // is supposed to evaluate " m2.colwise().sum()" into a temporary to avoid
  // evaluating the reduction multiple times
  if(ArrayType::RowsAtCompileTime>2 || ArrayType::RowsAtCompileTime==Dynamic)
  {
    m2.rowwise() /= m2.colwise().sum();
    VERIFY_IS_APPROX(m2, m1.rowwise() / m1.colwise().sum());
  }

  // all/any
  Array<bool,Dynamic,Dynamic> mb(rows,cols);
  mb = (m1.real()<=0.7).colwise().all();
  VERIFY( (mb.col(c) == (m1.real().col(c)<=0.7).all()).all() );
  mb = (m1.real()<=0.7).rowwise().all();
  VERIFY( (mb.row(r) == (m1.real().row(r)<=0.7).all()).all() );

  mb = (m1.real()>=0.7).colwise().any();
  VERIFY( (mb.col(c) == (m1.real().col(c)>=0.7).any()).all() );
  mb = (m1.real()>=0.7).rowwise().any();
  VERIFY( (mb.row(r) == (m1.real().row(r)>=0.7).any()).all() );
}

template<typename MatrixType> void vectorwiseop_matrix(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> ColVectorType;
  typedef Matrix<Scalar, 1, MatrixType::ColsAtCompileTime> RowVectorType;
  typedef Matrix<RealScalar, MatrixType::RowsAtCompileTime, 1> RealColVectorType;
  typedef Matrix<RealScalar, 1, MatrixType::ColsAtCompileTime> RealRowVectorType;
  typedef Matrix<Scalar,Dynamic,Dynamic> MatrixX;

  Index rows = m.rows();
  Index cols = m.cols();
  Index r = internal::random<Index>(0, rows-1),
        c = internal::random<Index>(0, cols-1);

  MatrixType m1 = MatrixType::Random(rows, cols),
            m2(rows, cols),
            m3(rows, cols);

  ColVectorType colvec = ColVectorType::Random(rows);
  RowVectorType rowvec = RowVectorType::Random(cols);
  RealColVectorType rcres;
  RealRowVectorType rrres;

  // test broadcast assignment
  m2 = m1;
  m2.colwise() = colvec;
  for(Index j=0; j<cols; ++j)
    VERIFY_IS_APPROX(m2.col(j), colvec);
  m2.rowwise() = rowvec;
  for(Index i=0; i<rows; ++i)
    VERIFY_IS_APPROX(m2.row(i), rowvec);
  if(rows>1)
    VERIFY_RAISES_ASSERT(m2.colwise() = colvec.transpose());
  if(cols>1)
    VERIFY_RAISES_ASSERT(m2.rowwise() = rowvec.transpose());

  // test addition

  m2 = m1;
  m2.colwise() += colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() + colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) + colvec);

  if(rows>1)
  {
    VERIFY_RAISES_ASSERT(m2.colwise() += colvec.transpose());
    VERIFY_RAISES_ASSERT(m1.colwise() + colvec.transpose());
  }

  m2 = m1;
  m2.rowwise() += rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() + rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) + rowvec);

  if(cols>1)
  {
    VERIFY_RAISES_ASSERT(m2.rowwise() += rowvec.transpose());
    VERIFY_RAISES_ASSERT(m1.rowwise() + rowvec.transpose());
  }

  // test substraction

  m2 = m1;
  m2.colwise() -= colvec;
  VERIFY_IS_APPROX(m2, m1.colwise() - colvec);
  VERIFY_IS_APPROX(m2.col(c), m1.col(c) - colvec);

  if(rows>1)
  {
    VERIFY_RAISES_ASSERT(m2.colwise() -= colvec.transpose());
    VERIFY_RAISES_ASSERT(m1.colwise() - colvec.transpose());
  }

  m2 = m1;
  m2.rowwise() -= rowvec;
  VERIFY_IS_APPROX(m2, m1.rowwise() - rowvec);
  VERIFY_IS_APPROX(m2.row(r), m1.row(r) - rowvec);

  if(cols>1)
  {
    VERIFY_RAISES_ASSERT(m2.rowwise() -= rowvec.transpose());
    VERIFY_RAISES_ASSERT(m1.rowwise() - rowvec.transpose());
  }

  // ------ partial reductions ------

  #define TEST_PARTIAL_REDUX_BASIC(FUNC,ROW,COL,PREPROCESS) {                          \
    ROW = m1 PREPROCESS .colwise().FUNC ;                                              \
    for(Index k=0; k<cols; ++k) VERIFY_IS_APPROX(ROW(k), m1.col(k) PREPROCESS .FUNC ); \
    COL = m1 PREPROCESS .rowwise().FUNC ;                                              \
    for(Index k=0; k<rows; ++k) VERIFY_IS_APPROX(COL(k), m1.row(k) PREPROCESS .FUNC ); \
  }

  TEST_PARTIAL_REDUX_BASIC(sum(),        rowvec,colvec,EIGEN_EMPTY);
  TEST_PARTIAL_REDUX_BASIC(prod(),       rowvec,colvec,EIGEN_EMPTY);
  TEST_PARTIAL_REDUX_BASIC(mean(),       rowvec,colvec,EIGEN_EMPTY);
  TEST_PARTIAL_REDUX_BASIC(minCoeff(),   rrres, rcres, .real());
  TEST_PARTIAL_REDUX_BASIC(maxCoeff(),   rrres, rcres, .real());
  TEST_PARTIAL_REDUX_BASIC(norm(),       rrres, rcres, EIGEN_EMPTY);
  TEST_PARTIAL_REDUX_BASIC(squaredNorm(),rrres, rcres, EIGEN_EMPTY);
  TEST_PARTIAL_REDUX_BASIC(redux(internal::scalar_sum_op<Scalar,Scalar>()),rowvec,colvec,EIGEN_EMPTY);

  VERIFY_IS_APPROX(m1.cwiseAbs().colwise().sum(), m1.colwise().template lpNorm<1>());
  VERIFY_IS_APPROX(m1.cwiseAbs().rowwise().sum(), m1.rowwise().template lpNorm<1>());
  VERIFY_IS_APPROX(m1.cwiseAbs().colwise().maxCoeff(), m1.colwise().template lpNorm<Infinity>());
  VERIFY_IS_APPROX(m1.cwiseAbs().rowwise().maxCoeff(), m1.rowwise().template lpNorm<Infinity>());

  // regression for bug 1158
  VERIFY_IS_APPROX(m1.cwiseAbs().colwise().sum().x(), m1.col(0).cwiseAbs().sum());

  // test normalized
  m2 = m1.colwise().normalized();
  VERIFY_IS_APPROX(m2.col(c), m1.col(c).normalized());
  m2 = m1.rowwise().normalized();
  VERIFY_IS_APPROX(m2.row(r), m1.row(r).normalized());

  // test normalize
  m2 = m1;
  m2.colwise().normalize();
  VERIFY_IS_APPROX(m2.col(c), m1.col(c).normalized());
  m2 = m1;
  m2.rowwise().normalize();
  VERIFY_IS_APPROX(m2.row(r), m1.row(r).normalized());

  // test with partial reduction of products
  Matrix<Scalar,MatrixType::RowsAtCompileTime,MatrixType::RowsAtCompileTime> m1m1 = m1 * m1.transpose();
  VERIFY_IS_APPROX( (m1 * m1.transpose()).colwise().sum(), m1m1.colwise().sum());
  Matrix<Scalar,1,MatrixType::RowsAtCompileTime> tmp(rows);
  VERIFY_EVALUATION_COUNT( tmp = (m1 * m1.transpose()).colwise().sum(), 1);

  m2 = m1.rowwise() - (m1.colwise().sum()/RealScalar(m1.rows())).eval();
  m1 = m1.rowwise() - (m1.colwise().sum()/RealScalar(m1.rows()));
  VERIFY_IS_APPROX( m1, m2 );
  VERIFY_EVALUATION_COUNT( m2 = (m1.rowwise() - m1.colwise().sum()/RealScalar(m1.rows())), (MatrixType::RowsAtCompileTime!=1 ? 1 : 0) );

  // test empty expressions
  VERIFY_IS_APPROX(m1.matrix().middleCols(0,0).rowwise().sum().eval(), MatrixX::Zero(rows,1));
  VERIFY_IS_APPROX(m1.matrix().middleRows(0,0).colwise().sum().eval(), MatrixX::Zero(1,cols));
  VERIFY_IS_APPROX(m1.matrix().middleCols(0,fix<0>).rowwise().sum().eval(), MatrixX::Zero(rows,1));
  VERIFY_IS_APPROX(m1.matrix().middleRows(0,fix<0>).colwise().sum().eval(), MatrixX::Zero(1,cols));

  VERIFY_IS_APPROX(m1.matrix().middleCols(0,0).rowwise().prod().eval(), MatrixX::Ones(rows,1));
  VERIFY_IS_APPROX(m1.matrix().middleRows(0,0).colwise().prod().eval(), MatrixX::Ones(1,cols));
  VERIFY_IS_APPROX(m1.matrix().middleCols(0,fix<0>).rowwise().prod().eval(), MatrixX::Ones(rows,1));
  VERIFY_IS_APPROX(m1.matrix().middleRows(0,fix<0>).colwise().prod().eval(), MatrixX::Ones(1,cols));
  
  VERIFY_IS_APPROX(m1.matrix().middleCols(0,0).rowwise().squaredNorm().eval(), MatrixX::Zero(rows,1));

  VERIFY_RAISES_ASSERT(m1.real().middleCols(0,0).rowwise().minCoeff().eval());
  VERIFY_RAISES_ASSERT(m1.real().middleRows(0,0).colwise().maxCoeff().eval());
  VERIFY_IS_EQUAL(m1.real().middleRows(0,0).rowwise().maxCoeff().eval().rows(),0);
  VERIFY_IS_EQUAL(m1.real().middleCols(0,0).colwise().maxCoeff().eval().cols(),0);
  VERIFY_IS_EQUAL(m1.real().middleRows(0,fix<0>).rowwise().maxCoeff().eval().rows(),0);
  VERIFY_IS_EQUAL(m1.real().middleCols(0,fix<0>).colwise().maxCoeff().eval().cols(),0);
}

EIGEN_DECLARE_TEST(vectorwiseop)
{
  CALL_SUBTEST_1( vectorwiseop_array(Array22cd()) );
  CALL_SUBTEST_2( vectorwiseop_array(Array<double, 3, 2>()) );
  CALL_SUBTEST_3( vectorwiseop_array(ArrayXXf(3, 4)) );
  CALL_SUBTEST_4( vectorwiseop_matrix(Matrix4cf()) );
  CALL_SUBTEST_5( vectorwiseop_matrix(Matrix4f()) );
  CALL_SUBTEST_5( vectorwiseop_matrix(Vector4f()) );
  CALL_SUBTEST_5( vectorwiseop_matrix(Matrix<float,4,5>()) );
  CALL_SUBTEST_6( vectorwiseop_matrix(MatrixXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE), internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  CALL_SUBTEST_7( vectorwiseop_matrix(VectorXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
  CALL_SUBTEST_7( vectorwiseop_matrix(RowVectorXd(internal::random<int>(1,EIGEN_TEST_MAX_SIZE))) );
}
