// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2009-2010 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_BLASUTIL_H
#define EIGEN_BLASUTIL_H

// This file contains many lightweight helper classes used to
// implement and control fast level 2 and level 3 BLAS-like routines.

namespace Eigen {

namespace internal {

// forward declarations
template<typename LhsScalar, typename RhsScalar, typename Index, typename DataMapper, int mr, int nr, bool ConjugateLhs=false, bool ConjugateRhs=false>
struct gebp_kernel;

template<typename Scalar, typename Index, typename DataMapper, int nr, int StorageOrder, bool Conjugate = false, bool PanelMode=false>
struct gemm_pack_rhs;

template<typename Scalar, typename Index, typename DataMapper, int Pack1, int Pack2, typename Packet, int StorageOrder, bool Conjugate = false, bool PanelMode = false>
struct gemm_pack_lhs;

template<
  typename Index,
  typename LhsScalar, int LhsStorageOrder, bool ConjugateLhs,
  typename RhsScalar, int RhsStorageOrder, bool ConjugateRhs,
  int ResStorageOrder, int ResInnerStride>
struct general_matrix_matrix_product;

template<typename Index,
         typename LhsScalar, typename LhsMapper, int LhsStorageOrder, bool ConjugateLhs,
         typename RhsScalar, typename RhsMapper, bool ConjugateRhs, int Version=Specialized>
struct general_matrix_vector_product;


template<bool Conjugate> struct conj_if;

template<> struct conj_if<true> {
  template<typename T>
  inline T operator()(const T& x) const { return numext::conj(x); }
  template<typename T>
  inline T pconj(const T& x) const { return internal::pconj(x); }
};

template<> struct conj_if<false> {
  template<typename T>
  inline const T& operator()(const T& x) const { return x; }
  template<typename T>
  inline const T& pconj(const T& x) const { return x; }
};

// Generic implementation for custom complex types.
template<typename LhsScalar, typename RhsScalar, bool ConjLhs, bool ConjRhs>
struct conj_helper
{
  typedef typename ScalarBinaryOpTraits<LhsScalar,RhsScalar>::ReturnType Scalar;

  EIGEN_STRONG_INLINE Scalar pmadd(const LhsScalar& x, const RhsScalar& y, const Scalar& c) const
  { return padd(c, pmul(x,y)); }

  EIGEN_STRONG_INLINE Scalar pmul(const LhsScalar& x, const RhsScalar& y) const
  { return conj_if<ConjLhs>()(x) *  conj_if<ConjRhs>()(y); }
};

template<typename Scalar> struct conj_helper<Scalar,Scalar,false,false>
{
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const { return internal::pmadd(x,y,c); }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const { return internal::pmul(x,y); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, false,true>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) + numext::imag(x)*numext::imag(y), numext::imag(x)*numext::real(y) - numext::real(x)*numext::imag(y)); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, true,false>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) + numext::imag(x)*numext::imag(y), numext::real(x)*numext::imag(y) - numext::imag(x)*numext::real(y)); }
};

template<typename RealScalar> struct conj_helper<std::complex<RealScalar>, std::complex<RealScalar>, true,true>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const Scalar& y, const Scalar& c) const
  { return c + pmul(x,y); }

  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const Scalar& y) const
  { return Scalar(numext::real(x)*numext::real(y) - numext::imag(x)*numext::imag(y), - numext::real(x)*numext::imag(y) - numext::imag(x)*numext::real(y)); }
};

template<typename RealScalar,bool Conj> struct conj_helper<std::complex<RealScalar>, RealScalar, Conj,false>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const Scalar& x, const RealScalar& y, const Scalar& c) const
  { return padd(c, pmul(x,y)); }
  EIGEN_STRONG_INLINE Scalar pmul(const Scalar& x, const RealScalar& y) const
  { return conj_if<Conj>()(x)*y; }
};

template<typename RealScalar,bool Conj> struct conj_helper<RealScalar, std::complex<RealScalar>, false,Conj>
{
  typedef std::complex<RealScalar> Scalar;
  EIGEN_STRONG_INLINE Scalar pmadd(const RealScalar& x, const Scalar& y, const Scalar& c) const
  { return padd(c, pmul(x,y)); }
  EIGEN_STRONG_INLINE Scalar pmul(const RealScalar& x, const Scalar& y) const
  { return x*conj_if<Conj>()(y); }
};

template<typename From,typename To> struct get_factor {
  EIGEN_DEVICE_FUNC static EIGEN_STRONG_INLINE To run(const From& x) { return To(x); }
};

template<typename Scalar> struct get_factor<Scalar,typename NumTraits<Scalar>::Real> {
  EIGEN_DEVICE_FUNC
  static EIGEN_STRONG_INLINE typename NumTraits<Scalar>::Real run(const Scalar& x) { return numext::real(x); }
};


template<typename Scalar, typename Index>
class BlasVectorMapper {
  public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE BlasVectorMapper(Scalar *data) : m_data(data) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar operator()(Index i) const {
    return m_data[i];
  }
  template <typename Packet, int AlignmentType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Packet load(Index i) const {
    return ploadt<Packet, AlignmentType>(m_data + i);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC bool aligned(Index i) const {
    return (UIntPtr(m_data+i)%sizeof(Packet))==0;
  }

  protected:
  Scalar* m_data;
};

template<typename Scalar, typename Index, int AlignmentType, int Incr=1>
class BlasLinearMapper;

template<typename Scalar, typename Index, int AlignmentType>
class BlasLinearMapper<Scalar,Index,AlignmentType>
{
public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE BlasLinearMapper(Scalar *data, Index incr=1)
    : m_data(data)
  {
    EIGEN_ONLY_USED_FOR_DEBUG(incr);
    eigen_assert(incr==1);
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void prefetch(int i) const {
    internal::prefetch(&operator()(i));
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar& operator()(Index i) const {
    return m_data[i];
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketType loadPacket(Index i) const {
    return ploadt<PacketType, AlignmentType>(m_data + i);
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void storePacket(Index i, const PacketType &p) const {
    pstoret<Scalar, PacketType, AlignmentType>(m_data + i, p);
  }

protected:
  Scalar *m_data;
};

// Lightweight helper class to access matrix coefficients.
template<typename Scalar, typename Index, int StorageOrder, int AlignmentType = Unaligned, int Incr = 1>
class blas_data_mapper;

template<typename Scalar, typename Index, int StorageOrder, int AlignmentType>
class blas_data_mapper<Scalar,Index,StorageOrder,AlignmentType,1>
{
public:
  typedef BlasLinearMapper<Scalar, Index, AlignmentType> LinearMapper;
  typedef BlasVectorMapper<Scalar, Index> VectorMapper;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE blas_data_mapper(Scalar* data, Index stride, Index incr=1)
   : m_data(data), m_stride(stride)
  {
    EIGEN_ONLY_USED_FOR_DEBUG(incr);
    eigen_assert(incr==1);
  }

  EIGEN_DEVICE_FUNC  EIGEN_ALWAYS_INLINE blas_data_mapper<Scalar, Index, StorageOrder, AlignmentType>
  getSubMapper(Index i, Index j) const {
    return blas_data_mapper<Scalar, Index, StorageOrder, AlignmentType>(&operator()(i, j), m_stride);
  }

  EIGEN_DEVICE_FUNC  EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(&operator()(i, j));
  }

  EIGEN_DEVICE_FUNC  EIGEN_ALWAYS_INLINE VectorMapper getVectorMapper(Index i, Index j) const {
    return VectorMapper(&operator()(i, j));
  }


  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar& operator()(Index i, Index j) const {
    return m_data[StorageOrder==RowMajor ? j + i*m_stride : i + j*m_stride];
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketType loadPacket(Index i, Index j) const {
    return ploadt<PacketType, AlignmentType>(&operator()(i, j));
  }

  template <typename PacketT, int AlignmentT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketT load(Index i, Index j) const {
    return ploadt<PacketT, AlignmentT>(&operator()(i, j));
  }

  template<typename SubPacket>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void scatterPacket(Index i, Index j, const SubPacket &p) const {
    pscatter<Scalar, SubPacket>(&operator()(i, j), p, m_stride);
  }

  template<typename SubPacket>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SubPacket gatherPacket(Index i, Index j) const {
    return pgather<Scalar, SubPacket>(&operator()(i, j), m_stride);
  }

  EIGEN_DEVICE_FUNC const Index stride() const { return m_stride; }
  EIGEN_DEVICE_FUNC const Scalar* data() const { return m_data; }

  EIGEN_DEVICE_FUNC Index firstAligned(Index size) const {
    if (UIntPtr(m_data)%sizeof(Scalar)) {
      return -1;
    }
    return internal::first_default_aligned(m_data, size);
  }

protected:
  Scalar* EIGEN_RESTRICT m_data;
  const Index m_stride;
};

// Implementation of non-natural increment (i.e. inner-stride != 1)
// The exposed API is not complete yet compared to the Incr==1 case
// because some features makes less sense in this case.
template<typename Scalar, typename Index, int AlignmentType, int Incr>
class BlasLinearMapper
{
public:
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE BlasLinearMapper(Scalar *data,Index incr) : m_data(data), m_incr(incr) {}

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void prefetch(int i) const {
    internal::prefetch(&operator()(i));
  }

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE Scalar& operator()(Index i) const {
    return m_data[i*m_incr.value()];
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketType loadPacket(Index i) const {
    return pgather<Scalar,PacketType>(m_data + i*m_incr.value(), m_incr.value());
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void storePacket(Index i, const PacketType &p) const {
    pscatter<Scalar, PacketType>(m_data + i*m_incr.value(), p, m_incr.value());
  }

protected:
  Scalar *m_data;
  const internal::variable_if_dynamic<Index,Incr> m_incr;
};

template<typename Scalar, typename Index, int StorageOrder, int AlignmentType,int Incr>
class blas_data_mapper
{
public:
  typedef BlasLinearMapper<Scalar, Index, AlignmentType,Incr> LinearMapper;

  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE blas_data_mapper(Scalar* data, Index stride, Index incr) : m_data(data), m_stride(stride), m_incr(incr) {}

  EIGEN_DEVICE_FUNC  EIGEN_ALWAYS_INLINE blas_data_mapper
  getSubMapper(Index i, Index j) const {
    return blas_data_mapper(&operator()(i, j), m_stride, m_incr.value());
  }

  EIGEN_DEVICE_FUNC  EIGEN_ALWAYS_INLINE LinearMapper getLinearMapper(Index i, Index j) const {
    return LinearMapper(&operator()(i, j), m_incr.value());
  }

  EIGEN_DEVICE_FUNC
  EIGEN_ALWAYS_INLINE Scalar& operator()(Index i, Index j) const {
    return m_data[StorageOrder==RowMajor ? j*m_incr.value() + i*m_stride : i*m_incr.value() + j*m_stride];
  }

  template<typename PacketType>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketType loadPacket(Index i, Index j) const {
    return pgather<Scalar,PacketType>(&operator()(i, j),m_incr.value());
  }

  template <typename PacketT, int AlignmentT>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE PacketT load(Index i, Index j) const {
    return pgather<Scalar,PacketT>(&operator()(i, j),m_incr.value());
  }

  template<typename SubPacket>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE void scatterPacket(Index i, Index j, const SubPacket &p) const {
    pscatter<Scalar, SubPacket>(&operator()(i, j), p, m_stride);
  }

  template<typename SubPacket>
  EIGEN_DEVICE_FUNC EIGEN_ALWAYS_INLINE SubPacket gatherPacket(Index i, Index j) const {
    return pgather<Scalar, SubPacket>(&operator()(i, j), m_stride);
  }

protected:
  Scalar* EIGEN_RESTRICT m_data;
  const Index m_stride;
  const internal::variable_if_dynamic<Index,Incr> m_incr;
};

// lightweight helper class to access matrix coefficients (const version)
template<typename Scalar, typename Index, int StorageOrder>
class const_blas_data_mapper : public blas_data_mapper<const Scalar, Index, StorageOrder> {
  public:
  EIGEN_ALWAYS_INLINE const_blas_data_mapper(const Scalar *data, Index stride) : blas_data_mapper<const Scalar, Index, StorageOrder>(data, stride) {}

  EIGEN_ALWAYS_INLINE const_blas_data_mapper<Scalar, Index, StorageOrder> getSubMapper(Index i, Index j) const {
    return const_blas_data_mapper<Scalar, Index, StorageOrder>(&(this->operator()(i, j)), this->m_stride);
  }
};


/* Helper class to analyze the factors of a Product expression.
 * In particular it allows to pop out operator-, scalar multiples,
 * and conjugate */
template<typename XprType> struct blas_traits
{
  typedef typename traits<XprType>::Scalar Scalar;
  typedef const XprType& ExtractType;
  typedef XprType _ExtractType;
  enum {
    IsComplex = NumTraits<Scalar>::IsComplex,
    IsTransposed = false,
    NeedToConjugate = false,
    HasUsableDirectAccess = (    (int(XprType::Flags)&DirectAccessBit)
                              && (   bool(XprType::IsVectorAtCompileTime)
                                  || int(inner_stride_at_compile_time<XprType>::ret) == 1)
                             ) ?  1 : 0,
    HasScalarFactor = false
  };
  typedef typename conditional<bool(HasUsableDirectAccess),
    ExtractType,
    typename _ExtractType::PlainObject
    >::type DirectLinearAccessType;
  static inline EIGEN_DEVICE_FUNC ExtractType extract(const XprType& x) { return x; }
  static inline EIGEN_DEVICE_FUNC const Scalar extractScalarFactor(const XprType&) { return Scalar(1); }
};

// pop conjugate
template<typename Scalar, typename NestedXpr>
struct blas_traits<CwiseUnaryOp<scalar_conjugate_op<Scalar>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseUnaryOp<scalar_conjugate_op<Scalar>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;

  enum {
    IsComplex = NumTraits<Scalar>::IsComplex,
    NeedToConjugate = Base::NeedToConjugate ? 0 : IsComplex
  };
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x) { return conj(Base::extractScalarFactor(x.nestedExpression())); }
};

// pop scalar multiple
template<typename Scalar, typename NestedXpr, typename Plain>
struct blas_traits<CwiseBinaryOp<scalar_product_op<Scalar>, const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  enum {
    HasScalarFactor = true
  };
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseBinaryOp<scalar_product_op<Scalar>, const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;
  static inline EIGEN_DEVICE_FUNC ExtractType extract(const XprType& x) { return Base::extract(x.rhs()); }
  static inline EIGEN_DEVICE_FUNC Scalar extractScalarFactor(const XprType& x)
  { return x.lhs().functor().m_other * Base::extractScalarFactor(x.rhs()); }
};
template<typename Scalar, typename NestedXpr, typename Plain>
struct blas_traits<CwiseBinaryOp<scalar_product_op<Scalar>, NestedXpr, const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain> > >
 : blas_traits<NestedXpr>
{
  enum {
    HasScalarFactor = true
  };
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseBinaryOp<scalar_product_op<Scalar>, NestedXpr, const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain> > XprType;
  typedef typename Base::ExtractType ExtractType;
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.lhs()); }
  static inline Scalar extractScalarFactor(const XprType& x)
  { return Base::extractScalarFactor(x.lhs()) * x.rhs().functor().m_other; }
};
template<typename Scalar, typename Plain1, typename Plain2>
struct blas_traits<CwiseBinaryOp<scalar_product_op<Scalar>, const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain1>,
                                                            const CwiseNullaryOp<scalar_constant_op<Scalar>,Plain2> > >
 : blas_traits<CwiseNullaryOp<scalar_constant_op<Scalar>,Plain1> >
{};

// pop opposite
template<typename Scalar, typename NestedXpr>
struct blas_traits<CwiseUnaryOp<scalar_opposite_op<Scalar>, NestedXpr> >
 : blas_traits<NestedXpr>
{
  enum {
    HasScalarFactor = true
  };
  typedef blas_traits<NestedXpr> Base;
  typedef CwiseUnaryOp<scalar_opposite_op<Scalar>, NestedXpr> XprType;
  typedef typename Base::ExtractType ExtractType;
  static inline ExtractType extract(const XprType& x) { return Base::extract(x.nestedExpression()); }
  static inline Scalar extractScalarFactor(const XprType& x)
  { return - Base::extractScalarFactor(x.nestedExpression()); }
};

// pop/push transpose
template<typename NestedXpr>
struct blas_traits<Transpose<NestedXpr> >
 : blas_traits<NestedXpr>
{
  typedef typename NestedXpr::Scalar Scalar;
  typedef blas_traits<NestedXpr> Base;
  typedef Transpose<NestedXpr> XprType;
  typedef Transpose<const typename Base::_ExtractType>  ExtractType; // const to get rid of a compile error; anyway blas traits are only used on the RHS
  typedef Transpose<const typename Base::_ExtractType> _ExtractType;
  typedef typename conditional<bool(Base::HasUsableDirectAccess),
    ExtractType,
    typename ExtractType::PlainObject
    >::type DirectLinearAccessType;
  enum {
    IsTransposed = Base::IsTransposed ? 0 : 1
  };
  static inline ExtractType extract(const XprType& x) { return ExtractType(Base::extract(x.nestedExpression())); }
  static inline Scalar extractScalarFactor(const XprType& x) { return Base::extractScalarFactor(x.nestedExpression()); }
};

template<typename T>
struct blas_traits<const T>
     : blas_traits<T>
{};

template<typename T, bool HasUsableDirectAccess=blas_traits<T>::HasUsableDirectAccess>
struct extract_data_selector {
  static const typename T::Scalar* run(const T& m)
  {
    return blas_traits<T>::extract(m).data();
  }
};

template<typename T>
struct extract_data_selector<T,false> {
  static typename T::Scalar* run(const T&) { return 0; }
};

template<typename T> const typename T::Scalar* extract_data(const T& m)
{
  return extract_data_selector<T>::run(m);
}

} // end namespace internal

} // end namespace Eigen

#endif // EIGEN_BLASUTIL_H
