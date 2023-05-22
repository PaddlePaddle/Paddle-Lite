// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Andy Davis <andydavis@google.com>
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#include "main.h"

#include <algorithm>
#include <set>

#include <Eigen/CXX11/Tensor>

using Eigen::Tensor;
using Eigen::Index;
using Eigen::RowMajor;
using Eigen::ColMajor;


template<typename T>
static const T& choose(int layout, const T& col, const T& row) {
  return layout == ColMajor ? col : row;
}

static internal::TensorBlockShapeType RandomShape() {
  return internal::random<bool>()
             ? internal::kUniformAllDims
             : internal::kSkewedInnerDims;
}

template <int NumDims>
static Index RandomTargetSize(const DSizes<Index, NumDims>& dims) {
  return internal::random<Index>(1, dims.TotalSize());
}

template <int NumDims>
static DSizes<Index, NumDims> RandomDims() {
  array<Index, NumDims> dims;
  for (int i = 0; i < NumDims; ++i) {
    dims[i] = internal::random<int>(1, 20);
  }
  return DSizes<Index, NumDims>(dims);
}

template <typename T>
static T* GenerateRandomData(const Index& size) {
  T* data = new T[size];
  for (int i = 0; i < size; ++i) {
    data[i] = internal::random<T>();
  }
  return data;
}

template <int NumDims>
static void Debug(DSizes<Index, NumDims> dims) {
  for (int i = 0; i < NumDims; ++i) {
    std::cout << dims[i] << "; ";
  }
  std::cout << std::endl;
}

template <int Layout>
static void test_block_mapper_sanity()
{
  typedef internal::TensorBlockMapper<int, Index, 2, Layout> TensorBlockMapper;

  DSizes<Index, 2> tensor_dims(100, 100);

  // Test uniform blocks.
  TensorBlockMapper uniform_block_mapper(
      tensor_dims, internal::kUniformAllDims, 100);

  VERIFY_IS_EQUAL(uniform_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(uniform_block_mapper.block_dims_total_size(), 100);

  // 10x10 blocks
  typename TensorBlockMapper::Block uniform_b0 = uniform_block_mapper.GetBlockForIndex(0, NULL);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(0), 10);
  VERIFY_IS_EQUAL(uniform_b0.block_sizes().at(1), 10);
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(0), choose(Layout, 1, 10));
  VERIFY_IS_EQUAL(uniform_b0.block_strides().at(1), choose(Layout, 10, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(uniform_b0.tensor_strides().at(1), choose(Layout, 100, 1));

  // Test skewed to inner dims blocks.
  TensorBlockMapper skewed_block_mapper(
      tensor_dims, internal::kSkewedInnerDims, 100);

  VERIFY_IS_EQUAL(skewed_block_mapper.total_block_count(), 100);
  VERIFY_IS_EQUAL(skewed_block_mapper.block_dims_total_size(), 100);

  // 1x100 (100x1) rows/cols depending on a tensor layout.
  typename TensorBlockMapper::Block skewed_b0 = skewed_block_mapper.GetBlockForIndex(0, NULL);
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(0), choose(Layout, 100, 1));
  VERIFY_IS_EQUAL(skewed_b0.block_sizes().at(1), choose(Layout, 1, 100));
  // Depending on a layout we stride by cols rows.
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.block_strides().at(1), choose(Layout, 100, 1));
  // Tensor strides depend only on a layout and not on the block size.
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(0), choose(Layout, 1, 100));
  VERIFY_IS_EQUAL(skewed_b0.tensor_strides().at(1), choose(Layout, 100, 1));
}

// Given a TensorBlock "visit" every element accessible though it, and a keep an
// index in the visited set. Verify that every coeff accessed only once.
template <typename T, int Layout, int NumDims>
static void UpdateCoeffSet(
    const internal::TensorBlock<T, Index, NumDims, Layout>& block,
    Index first_coeff_index, int dim_index, std::set<Index>* visited_coeffs) {
  const DSizes<Index, NumDims>& block_sizes = block.block_sizes();
  const DSizes<Index, NumDims>& tensor_strides = block.tensor_strides();

  for (int i = 0; i < block_sizes[dim_index]; ++i) {
    if (tensor_strides[dim_index] == 1) {
      typedef std::pair<std::set<Index>::iterator, bool> ReturnType;
      ReturnType inserted = visited_coeffs->insert(first_coeff_index + i);
      VERIFY_IS_EQUAL(inserted.second, true);
    } else {
      int next_dim_index = dim_index + choose(Layout, -1, 1);
      UpdateCoeffSet<T, Layout, NumDims>(block, first_coeff_index,
                                         next_dim_index, visited_coeffs);
      first_coeff_index += tensor_strides[dim_index];
    }
  }
}

template <typename T, int NumDims, int Layout>
static void test_block_mapper_maps_every_element() {
  typedef internal::TensorBlock<T, Index, NumDims, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<T, Index, NumDims, Layout> TensorBlockMapper;

  DSizes<Index, NumDims> dims = RandomDims<NumDims>();

  // Keep track of elements indices available via block access.
  std::set<Index> coeff_set;

  // Try different combinations of block types and sizes.
  TensorBlockMapper block_mapper(dims, RandomShape(), RandomTargetSize(dims));

  for (int i = 0; i < block_mapper.total_block_count(); ++i) {
    TensorBlock block = block_mapper.GetBlockForIndex(i, NULL);
    UpdateCoeffSet<T, Layout, NumDims>(block, block.first_coeff_index(),
                                       choose(Layout, NumDims - 1, 0),
                                       &coeff_set);
  }

  // Verify that every coefficient in the original Tensor is accessible through
  // TensorBlock only once.
  Index total_coeffs = dims.TotalSize();
  VERIFY_IS_EQUAL(Index(coeff_set.size()), total_coeffs);
  VERIFY_IS_EQUAL(*coeff_set.begin(), 0);
  VERIFY_IS_EQUAL(*coeff_set.rbegin(), total_coeffs - 1);
}

template <int Layout, int NumDims>
static Index GetInputIndex(Index output_index,
                         const array<Index, NumDims>& output_to_input_dim_map,
                         const array<Index, NumDims>& input_strides,
                         const array<Index, NumDims>& output_strides) {
  int input_index = 0;
  if (Layout == ColMajor) {
    for (int i = NumDims - 1; i > 0; --i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[0]];
  } else {
    for (int i = 0; i < NumDims - 1; ++i) {
      const Index idx = output_index / output_strides[i];
      input_index += idx * input_strides[output_to_input_dim_map[i]];
      output_index -= idx * output_strides[i];
    }
    return input_index +
           output_index * input_strides[output_to_input_dim_map[NumDims - 1]];
  }
}

template <int Layout, int NumDims>
static array<Index, NumDims> ComputeStrides(
    const array<Index, NumDims>& sizes) {
  array<Index, NumDims> strides;
  if (Layout == ColMajor) {
    strides[0] = 1;
    for (int i = 1; i < NumDims; ++i) {
      strides[i] = strides[i - 1] * sizes[i - 1];
    }
  } else {
    strides[NumDims - 1] = 1;
    for (int i = NumDims - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * sizes[i + 1];
    }
  }
  return strides;
}

template<typename Scalar, typename StorageIndex, int Dim>
class EqualityChecker
{
    const Scalar* input_data;
    const DSizes<StorageIndex, Dim> &input_dims, &input_strides, &output_dims, &output_strides;
    void check_recursive(const Scalar* input, const Scalar* output, int depth=0) const
    {
        if(depth==Dim)
        {
            VERIFY_IS_EQUAL(*input, *output);
            return;
        }

        for(int i=0; i<output_dims[depth]; ++i)
        {
            check_recursive(input + i % input_dims[depth] * input_strides[depth], output + i*output_strides[depth], depth+1);
        }
    }
public:
    EqualityChecker(const Scalar* input_data_,
            const DSizes<StorageIndex, Dim> &input_dims_, const DSizes<StorageIndex, Dim> &input_strides_,
            const DSizes<StorageIndex, Dim> &output_dims_, const DSizes<StorageIndex, Dim> &output_strides_)
        : input_data(input_data_)
        , input_dims(input_dims_), input_strides(input_strides_)
        , output_dims(output_dims_), output_strides(output_strides_)
        {}

    void operator()(const Scalar* output_data) const
    {
        check_recursive(input_data, output_data);
    }
};

template <int Layout>
static void test_uniform_block_shape()
{
  typedef internal::TensorBlock<int, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<int, Index, 5, Layout> TensorBlockMapper;

  {
    // Test shape 'UniformAllDims' with uniform 'max_coeff count'.
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    for (int i = 0; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // partially into first inner-most dimension.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 6;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(6, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // fully into first inner-most dimension.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 5 * 5 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 5 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(5, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with larger 'max_coeff count' which spills
  // fully into first few inner-most dimensions.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(7, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 7 * 5;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(7, 5, 6, 9, 7);
    const Index max_coeff_count = 5 * 5 * 5 * 6 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'UniformAllDims' with full allocation to all dims.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(7, 5, 6, 17, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(7, 5, 6, 9, 7);
    const Index max_coeff_count = 7 * 5 * 6 * 9 * 7;
    TensorBlockMapper block_mapper(dims, internal::kUniformAllDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(9, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }
}

template <int Layout>
static void test_skewed_inner_dim_block_shape()
{
  typedef internal::TensorBlock<int, Index, 5, Layout> TensorBlock;
  typedef internal::TensorBlockMapper<int, Index, 5, Layout> TensorBlockMapper;

  // Test shape 'SkewedInnerDims' with partial allocation to inner-most dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 10 * 1 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(10, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 1 * 6;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(6, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 1 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    for (int i = 1; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 1 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    for (int i = 3; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim,
  // and partial allocation to second inner-dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 3 * 1 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(3, block.block_sizes()[1]);
    for (int i = 2; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 1 * 15 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(15, block.block_sizes()[3]);
    for (int i = 2; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to inner-most dim,
  // and partial allocation to third inner-dim.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 5 * 1 * 1;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    for (int i = 3; i < 5; ++i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 1 * 1 * 5 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[2]);
    for (int i = 1; i >= 0; --i) {
      VERIFY_IS_EQUAL(1, block.block_sizes()[i]);
    }
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }

  // Test shape 'SkewedInnerDims' with full allocation to all dims.
  if (Layout == ColMajor) {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  } else {
    DSizes<Index, 5> dims(11, 5, 6, 17, 7);
    const Index max_coeff_count = 11 * 5 * 6 * 17 * 7;
    TensorBlockMapper block_mapper(dims, internal::kSkewedInnerDims,
                                   max_coeff_count);
    TensorBlock block = block_mapper.GetBlockForIndex(0, NULL);
    VERIFY_IS_EQUAL(7, block.block_sizes()[4]);
    VERIFY_IS_EQUAL(17, block.block_sizes()[3]);
    VERIFY_IS_EQUAL(6, block.block_sizes()[2]);
    VERIFY_IS_EQUAL(5, block.block_sizes()[1]);
    VERIFY_IS_EQUAL(11, block.block_sizes()[0]);
    VERIFY(block.block_sizes().TotalSize() <= max_coeff_count);
  }
}

template <int Layout>
static void test_empty_dims(const internal::TensorBlockShapeType block_shape)
{
  // Test blocking of tensors with zero dimensions:
  //  - we must not crash on asserts and divisions by zero
  //  - we must not return block with zero dimensions
  //    (recipe for overflows/underflows, divisions by zero and NaNs later)
  //  - total block count must be zero
  {
    typedef internal::TensorBlockMapper<int, Index, 1, Layout> TensorBlockMapper;
    DSizes<Index, 1> dims(0);
    for (int max_coeff_count = 0; max_coeff_count < 2; ++max_coeff_count) {
      TensorBlockMapper block_mapper(dims, block_shape, max_coeff_count);
      VERIFY_IS_EQUAL(block_mapper.total_block_count(), 0);
      VERIFY(block_mapper.block_dims_total_size() >= 1);
    }
  }

  {
    typedef internal::TensorBlockMapper<int, Index, 2, Layout> TensorBlockMapper;
    for (int dim1 = 0; dim1 < 3; ++dim1) {
      for (int dim2 = 0; dim2 < 3; ++dim2) {
        DSizes<Index, 2> dims(dim1, dim2);
        for (int max_coeff_count = 0; max_coeff_count < 2; ++max_coeff_count) {
          TensorBlockMapper block_mapper(dims, block_shape, max_coeff_count);
          if (dim1 * dim2 == 0) {
            VERIFY_IS_EQUAL(block_mapper.total_block_count(), 0);
          }
          VERIFY(block_mapper.block_dims_total_size() >= 1);
        }
      }
    }
  }
}

#define TEST_LAYOUTS(NAME) \
  CALL_SUBTEST(NAME<ColMajor>()); \
  CALL_SUBTEST(NAME<RowMajor>())

#define TEST_LAYOUTS_AND_DIMS(TYPE, NAME)    \
  CALL_SUBTEST((NAME<TYPE, 1, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 1, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 2, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 2, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 3, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 3, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 4, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 4, RowMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 5, ColMajor>())); \
  CALL_SUBTEST((NAME<TYPE, 5, RowMajor>()))

#define TEST_LAYOUTS_WITH_ARG(NAME, ARG) \
  CALL_SUBTEST(NAME<ColMajor>(ARG)); \
  CALL_SUBTEST(NAME<RowMajor>(ARG))

EIGEN_DECLARE_TEST(cxx11_tensor_block_access) {
  TEST_LAYOUTS(test_block_mapper_sanity);
  TEST_LAYOUTS_AND_DIMS(float, test_block_mapper_maps_every_element);
  TEST_LAYOUTS(test_uniform_block_shape);
  TEST_LAYOUTS(test_skewed_inner_dim_block_shape);
  TEST_LAYOUTS_WITH_ARG(test_empty_dims, internal::kUniformAllDims);
  TEST_LAYOUTS_WITH_ARG(test_empty_dims, internal::kSkewedInnerDims);
}

#undef TEST_LAYOUTS
#undef TEST_LAYOUTS_WITH_ARG
