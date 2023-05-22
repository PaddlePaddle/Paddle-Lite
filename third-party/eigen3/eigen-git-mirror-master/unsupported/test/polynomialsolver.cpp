// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2010 Manuel Yguel <manuel.yguel@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"
#include <unsupported/Eigen/Polynomials>
#include <iostream>
#include <algorithm>

using namespace std;

namespace Eigen {
namespace internal {
template<int Size>
struct increment_if_fixed_size
{
  enum {
    ret = (Size == Dynamic) ? Dynamic : Size+1
  };
};
}
}

template<typename PolynomialType>
PolynomialType polyder(const PolynomialType& p)
{
  typedef typename PolynomialType::Scalar Scalar;
  PolynomialType res(p.size());
  for(Index i=1; i<p.size(); ++i)
    res[i-1] = p[i]*Scalar(i);
  res[p.size()-1] = 0.;
  return res;
}

template<int Deg, typename POLYNOMIAL, typename SOLVER>
bool aux_evalSolver( const POLYNOMIAL& pols, SOLVER& psolve )
{
  typedef typename POLYNOMIAL::Scalar Scalar;
  typedef typename POLYNOMIAL::RealScalar RealScalar;

  typedef typename SOLVER::RootsType    RootsType;
  typedef Matrix<RealScalar,Deg,1>      EvalRootsType;

  const Index deg = pols.size()-1;

  // Test template constructor from coefficient vector
  SOLVER solve_constr (pols);

  psolve.compute( pols );
  const RootsType& roots( psolve.roots() );
  EvalRootsType evr( deg );
  POLYNOMIAL pols_der = polyder(pols);
  EvalRootsType der( deg );
  for( int i=0; i<roots.size(); ++i ){
    evr[i] = std::abs( poly_eval( pols, roots[i] ) );
    der[i] = numext::maxi(RealScalar(1.), std::abs( poly_eval( pols_der, roots[i] ) ));
  }

  // we need to divide by the magnitude of the derivative because
  // with a high derivative is very small error in the value of the root
  // yiels a very large error in the polynomial evaluation.
  bool evalToZero = (evr.cwiseQuotient(der)).isZero( test_precision<Scalar>() );
  if( !evalToZero )
  {
    cerr << "WRONG root: " << endl;
    cerr << "Polynomial: " << pols.transpose() << endl;
    cerr << "Roots found: " << roots.transpose() << endl;
    cerr << "Abs value of the polynomial at the roots: " << evr.transpose() << endl;
    cerr << endl;
  }

  std::vector<RealScalar> rootModuli( roots.size() );
  Map< EvalRootsType > aux( &rootModuli[0], roots.size() );
  aux = roots.array().abs();
  std::sort( rootModuli.begin(), rootModuli.end() );
  bool distinctModuli=true;
  for( size_t i=1; i<rootModuli.size() && distinctModuli; ++i )
  {
    if( internal::isApprox( rootModuli[i], rootModuli[i-1] ) ){
      distinctModuli = false; }
  }
  VERIFY( evalToZero || !distinctModuli );

  return distinctModuli;
}







template<int Deg, typename POLYNOMIAL>
void evalSolver( const POLYNOMIAL& pols )
{
  typedef typename POLYNOMIAL::Scalar Scalar;

  typedef PolynomialSolver<Scalar, Deg > PolynomialSolverType;

  PolynomialSolverType psolve;
  aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>( pols, psolve );
}




template< int Deg, typename POLYNOMIAL, typename ROOTS, typename REAL_ROOTS >
void evalSolverSugarFunction( const POLYNOMIAL& pols, const ROOTS& roots, const REAL_ROOTS& real_roots )
{
  using std::sqrt;
  typedef typename POLYNOMIAL::Scalar Scalar;
  typedef typename POLYNOMIAL::RealScalar RealScalar;

  typedef PolynomialSolver<Scalar, Deg >              PolynomialSolverType;

  PolynomialSolverType psolve;
  if( aux_evalSolver<Deg, POLYNOMIAL, PolynomialSolverType>( pols, psolve ) )
  {
    //It is supposed that
    // 1) the roots found are correct
    // 2) the roots have distinct moduli

    //Test realRoots
    std::vector< RealScalar > calc_realRoots;
    psolve.realRoots( calc_realRoots,  test_precision<RealScalar>());
    VERIFY_IS_EQUAL( calc_realRoots.size() , (size_t)real_roots.size() );

    const RealScalar psPrec = sqrt( test_precision<RealScalar>() );

    for( size_t i=0; i<calc_realRoots.size(); ++i )
    {
      bool found = false;
      for( size_t j=0; j<calc_realRoots.size()&& !found; ++j )
      {
        if( internal::isApprox( calc_realRoots[i], real_roots[j], psPrec ) ){
          found = true; }
      }
      VERIFY( found );
    }

    //Test greatestRoot
    VERIFY( internal::isApprox( roots.array().abs().maxCoeff(),
          abs( psolve.greatestRoot() ), psPrec ) );

    //Test smallestRoot
    VERIFY( internal::isApprox( roots.array().abs().minCoeff(),
          abs( psolve.smallestRoot() ), psPrec ) );

    bool hasRealRoot;
    //Test absGreatestRealRoot
    RealScalar r = psolve.absGreatestRealRoot( hasRealRoot );
    VERIFY( hasRealRoot == (real_roots.size() > 0 ) );
    if( hasRealRoot ){
      VERIFY( internal::isApprox( real_roots.array().abs().maxCoeff(), abs(r), psPrec ) );  }

    //Test absSmallestRealRoot
    r = psolve.absSmallestRealRoot( hasRealRoot );
    VERIFY( hasRealRoot == (real_roots.size() > 0 ) );
    if( hasRealRoot ){
      VERIFY( internal::isApprox( real_roots.array().abs().minCoeff(), abs( r ), psPrec ) ); }

    //Test greatestRealRoot
    r = psolve.greatestRealRoot( hasRealRoot );
    VERIFY( hasRealRoot == (real_roots.size() > 0 ) );
    if( hasRealRoot ){
      VERIFY( internal::isApprox( real_roots.array().maxCoeff(), r, psPrec ) ); }

    //Test smallestRealRoot
    r = psolve.smallestRealRoot( hasRealRoot );
    VERIFY( hasRealRoot == (real_roots.size() > 0 ) );
    if( hasRealRoot ){
    VERIFY( internal::isApprox( real_roots.array().minCoeff(), r, psPrec ) ); }
  }
}


template<typename _Scalar, int _Deg>
void polynomialsolver(int deg)
{
  typedef typename NumTraits<_Scalar>::Real RealScalar;
  typedef internal::increment_if_fixed_size<_Deg>     Dim;
  typedef Matrix<_Scalar,Dim::ret,1>                  PolynomialType;
  typedef Matrix<_Scalar,_Deg,1>                      EvalRootsType;
  typedef Matrix<RealScalar,_Deg,1>                   RealRootsType;

  cout << "Standard cases" << endl;
  PolynomialType pols = PolynomialType::Random(deg+1);
  evalSolver<_Deg,PolynomialType>( pols );

  cout << "Hard cases" << endl;
  _Scalar multipleRoot = internal::random<_Scalar>();
  EvalRootsType allRoots = EvalRootsType::Constant(deg,multipleRoot);
  roots_to_monicPolynomial( allRoots, pols );
  evalSolver<_Deg,PolynomialType>( pols );

  cout << "Test sugar" << endl;
  RealRootsType realRoots = RealRootsType::Random(deg);
  roots_to_monicPolynomial( realRoots, pols );
  evalSolverSugarFunction<_Deg>(
      pols,
      realRoots.template cast <std::complex<RealScalar> >().eval(),
      realRoots );
}

EIGEN_DECLARE_TEST(polynomialsolver)
{
  for(int i = 0; i < g_repeat; i++)
  {
    CALL_SUBTEST_1( (polynomialsolver<float,1>(1)) );
    CALL_SUBTEST_2( (polynomialsolver<double,2>(2)) );
    CALL_SUBTEST_3( (polynomialsolver<double,3>(3)) );
    CALL_SUBTEST_4( (polynomialsolver<float,4>(4)) );
    CALL_SUBTEST_5( (polynomialsolver<double,5>(5)) );
    CALL_SUBTEST_6( (polynomialsolver<float,6>(6)) );
    CALL_SUBTEST_7( (polynomialsolver<float,7>(7)) );
    CALL_SUBTEST_8( (polynomialsolver<double,8>(8)) );

    CALL_SUBTEST_9( (polynomialsolver<float,Dynamic>(
            internal::random<int>(9,13)
            )) );
    CALL_SUBTEST_10((polynomialsolver<double,Dynamic>(
            internal::random<int>(9,13)
            )) );
    CALL_SUBTEST_11((polynomialsolver<float,Dynamic>(1)) );
    CALL_SUBTEST_12((polynomialsolver<std::complex<double>,Dynamic>(internal::random<int>(2,13))) );
  }
}
