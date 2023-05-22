// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2018 Andy Davis <andydavis@google.com>
// Copyright (C) 2018 Eugene Zhulenev <ezhulenev@google.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
#define EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H

namespace Eigen {
namespace internal {

namespace {

// Helper template to choose between ColMajor and RowMajor values.
template <int Layout>
struct cond;

template <>
struct cond<ColMajor> {
  template <typename T>
  EIGEN_STRONG_INLINE const T& operator()(const T& col,
                                          const T& /*row*/) const {
    return col;
  }
};

template <>
struct cond<RowMajor> {
  template <typename T>
  EIGEN_STRONG_INLINE const T& operator()(const T& /*col*/,
                                          const T& row) const {
    return row;
  }
};

}  // namespace

/**
 * \enum TensorBlockShapeType
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block shape type.
 *
 * Tensor block shape type defines what are the shape preference for the blocks
 * extracted from the larger tensor.
 *
 * Example:
 *
 * We want to extract blocks of 100 elements from the large 100x100 tensor:
 *  - tensor: 100x100
 *  - target_block_size: 100
 *
 * TensorBlockShapeType:
 *  - kUniformAllDims: 100 blocks of size 10x10
 *  - kSkewedInnerDims: 100 blocks of size 100x1 (or 1x100 depending on a column
 *                      or row major layout)
 */
enum TensorBlockShapeType {
  kUniformAllDims,
  kSkewedInnerDims
};

struct TensorOpResourceRequirements {
  TensorBlockShapeType block_shape;
  Index block_total_size;
  // TODO(andydavis) Add 'target_num_threads' to support communication of
  // thread-resource requirements. This will allow ops deep in the
  // expression tree (like reductions) to communicate resources
  // requirements based on local state (like the total number of reductions
  // to be computed).
  TensorOpResourceRequirements(TensorBlockShapeType shape,
                               const Index size)
      : block_shape(shape), block_total_size(size) {}
};

// Tries to merge multiple resource requirements.
EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void MergeResourceRequirements(
    const std::vector<TensorOpResourceRequirements>& resources,
    TensorBlockShapeType* block_shape, Index* block_total_size) {
  if (resources.empty()) {
    return;
  }
  // TODO(andydavis) Implement different policies (i.e. revert to a default
  // policy if block shapes/sizes conflict).
  *block_shape = resources[0].block_shape;
  *block_total_size = resources[0].block_total_size;
  for (std::vector<TensorOpResourceRequirements>::size_type i = 1; i < resources.size(); ++i) {
    if (resources[i].block_shape == kSkewedInnerDims &&
        *block_shape != kSkewedInnerDims) {
      *block_shape = kSkewedInnerDims;
    }
    *block_total_size =
        numext::maxi(*block_total_size, resources[i].block_total_size);
  }
}

/**
 * \class TensorBlock
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block class.
 *
 * This class represents a tensor block specified by the index of the
 * first block coefficient, and the size of the block in each dimension.
 */
template <typename Scalar, typename StorageIndex, int NumDims, int Layout>
class TensorBlock {
 public:
  typedef DSizes<StorageIndex, NumDims> Dimensions;

  TensorBlock(const StorageIndex first_coeff_index, const Dimensions& block_sizes,
              const Dimensions& block_strides, const Dimensions& tensor_strides,
              Scalar* data)
      : m_first_coeff_index(first_coeff_index),
        m_block_sizes(block_sizes),
        m_block_strides(block_strides),
        m_tensor_strides(tensor_strides),
        m_data(data) {}

  StorageIndex first_coeff_index() const { return m_first_coeff_index; }

  const Dimensions& block_sizes() const { return m_block_sizes; }

  const Dimensions& block_strides() const { return m_block_strides; }

  const Dimensions& tensor_strides() const { return m_tensor_strides; }

  Scalar* data() { return m_data; }

  const Scalar* data() const { return m_data; }

 private:
  StorageIndex m_first_coeff_index;
  Dimensions m_block_sizes;
  Dimensions m_block_strides;
  Dimensions m_tensor_strides;
  Scalar* m_data;  // Not owned.
};

/**
 * \class TensorBlockMapper
 * \ingroup CXX11_Tensor_Module
 *
 * \brief Tensor block mapper class.
 *
 * This class is responsible for iterating over the blocks of a tensor.
 */
template <typename Scalar, typename StorageIndex, int NumDims, int Layout>
class TensorBlockMapper {
 public:
  typedef TensorBlock<Scalar, StorageIndex, NumDims, Layout> Block;
  typedef DSizes<StorageIndex, NumDims> Dimensions;

  TensorBlockMapper() {}
  TensorBlockMapper(const Dimensions& dims,
                    const TensorBlockShapeType block_shape,
                    Index min_target_size)
      : m_dimensions(dims),
        m_block_dim_sizes(BlockDimensions(dims, block_shape, convert_index<StorageIndex>(min_target_size))) {
    // Calculate block counts by dimension and total block count.
    DSizes<StorageIndex, NumDims> block_count;
    for (Index i = 0; i < block_count.rank(); ++i) {
      block_count[i] = divup(m_dimensions[i], m_block_dim_sizes[i]);
    }
    m_total_block_count = array_prod(block_count);

    // Calculate block strides (used for enumerating blocks).
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        m_block_strides[0] = 1;
        m_tensor_strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          m_block_strides[i] = m_block_strides[i - 1] * block_count[i - 1];
          m_tensor_strides[i] = m_tensor_strides[i - 1] * m_dimensions[i - 1];
        }
      } else {
        m_block_strides[NumDims - 1] = 1;
        m_tensor_strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          m_block_strides[i] = m_block_strides[i + 1] * block_count[i + 1];
          m_tensor_strides[i] = m_tensor_strides[i + 1] * m_dimensions[i + 1];
        }
      }
    }
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Block
  GetBlockForIndex(StorageIndex block_index, Scalar* data) const {
    StorageIndex first_coeff_index = 0;
    DSizes<StorageIndex, NumDims> coords;
    DSizes<StorageIndex, NumDims> sizes;
    DSizes<StorageIndex, NumDims> strides;
    if (NumDims > 0) {
      if (static_cast<int>(Layout) == static_cast<int>(ColMajor)) {
        for (int i = NumDims - 1; i > 0; --i) {
          const StorageIndex idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[0] = block_index * m_block_dim_sizes[0];
        sizes[0] =
            numext::mini((m_dimensions[0] - coords[0]), m_block_dim_sizes[0]);
        first_coeff_index += coords[0] * m_tensor_strides[0];

        strides[0] = 1;
        for (int i = 1; i < NumDims; ++i) {
          strides[i] = strides[i - 1] * sizes[i - 1];
        }
      } else {
        for (int i = 0; i < NumDims - 1; ++i) {
          const StorageIndex idx = block_index / m_block_strides[i];
          coords[i] = idx * m_block_dim_sizes[i];
          sizes[i] =
              numext::mini((m_dimensions[i] - coords[i]), m_block_dim_sizes[i]);
          block_index -= idx * m_block_strides[i];
          first_coeff_index += coords[i] * m_tensor_strides[i];
        }
        coords[NumDims - 1] = block_index * m_block_dim_sizes[NumDims - 1];
        sizes[NumDims - 1] =
            numext::mini((m_dimensions[NumDims - 1] - coords[NumDims - 1]),
                         m_block_dim_sizes[NumDims - 1]);
        first_coeff_index +=
            coords[NumDims - 1] * m_tensor_strides[NumDims - 1];

        strides[NumDims - 1] = 1;
        for (int i = NumDims - 2; i >= 0; --i) {
          strides[i] = strides[i + 1] * sizes[i + 1];
        }
      }
    }

    return Block(first_coeff_index, sizes, strides, m_tensor_strides, data);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE StorageIndex total_block_count() const {
    return m_total_block_count;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE StorageIndex
  block_dims_total_size() const {
    return m_block_dim_sizes.TotalSize();
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE const Dimensions&
  block_dim_sizes() const {
    return m_block_dim_sizes;
  }

 private:
  static Dimensions BlockDimensions(const Dimensions& tensor_dims,
                                    const TensorBlockShapeType block_shape,
                                    StorageIndex min_target_size) {
    min_target_size = numext::maxi<StorageIndex>(1, min_target_size);

    // If tensor fully fits into the target size, we'll treat it a single block.
    Dimensions block_dim_sizes = tensor_dims;

    if (tensor_dims.TotalSize() == 0) {
      // Corner case: one of the dimensions is zero. Logic below is too complex
      // to handle this case on a general basis, just use unit block size.
      // Note: we must not yield blocks with zero dimensions (recipe for
      // overflows/underflows, divisions by zero and NaNs later).
      for (int i = 0; i < NumDims; ++i) {
        block_dim_sizes[i] = 1;
      }
    } else if (block_dim_sizes.TotalSize() > min_target_size) {
      if (block_shape == kUniformAllDims) {
        // Tensor will not fit within 'min_target_size' budget: calculate tensor
        // block dimension sizes based on "square" dimension size target.
        const StorageIndex dim_size_target = convert_index<StorageIndex>(
          std::pow(static_cast<float>(min_target_size),
                   1.0f / static_cast<float>(block_dim_sizes.rank())));
        for (Index i = 0; i < block_dim_sizes.rank(); ++i) {
          // TODO(andydavis) Adjust the inner most 'block_dim_size' to make it
          // a multiple of the packet size. Note that reducing
          // 'block_dim_size' in this manner can increase the number of
          // blocks, and so will amplify any per-block overhead.
          block_dim_sizes[i] = numext::mini(dim_size_target, tensor_dims[i]);
        }
        // Add any un-allocated coefficients to inner dimension(s).
        StorageIndex total_size = block_dim_sizes.TotalSize();
        for (int i = 0; i < NumDims; ++i) {
          const int dim = cond<Layout>()(i, NumDims - i - 1);
          if (block_dim_sizes[dim] < tensor_dims[dim]) {
            const StorageIndex total_size_other_dims =
                total_size / block_dim_sizes[dim];
            const StorageIndex alloc_avail =
                divup<StorageIndex>(min_target_size, total_size_other_dims);
            if (alloc_avail == block_dim_sizes[dim]) {
              // Insufficient excess coefficients to allocate.
              break;
            }
            block_dim_sizes[dim] = numext::mini(tensor_dims[dim], alloc_avail);
            total_size = total_size_other_dims * block_dim_sizes[dim];
          }
        }
      } else if (block_shape == kSkewedInnerDims) {
        StorageIndex coeff_to_allocate = min_target_size;
        for (int i = 0; i < NumDims; ++i) {
          const int dim = cond<Layout>()(i, NumDims - i - 1);
          block_dim_sizes[dim] =
              numext::mini(coeff_to_allocate, tensor_dims[dim]);
          coeff_to_allocate = divup(
              coeff_to_allocate,
              numext::maxi(static_cast<StorageIndex>(1), block_dim_sizes[dim]));
        }
        eigen_assert(coeff_to_allocate == 1);
      } else {
        eigen_assert(false);  // someone added new block shape type
      }
    }

    eigen_assert(
        block_dim_sizes.TotalSize() >=
        numext::mini<Index>(min_target_size, tensor_dims.TotalSize()));

    return block_dim_sizes;
  }

  Dimensions m_dimensions;
  Dimensions m_block_dim_sizes;
  Dimensions m_block_strides;
  Dimensions m_tensor_strides;
  StorageIndex m_total_block_count;
};

}  // namespace internal

}  // namespace Eigen

#endif  // EIGEN_CXX11_TENSOR_TENSOR_BLOCK_H
