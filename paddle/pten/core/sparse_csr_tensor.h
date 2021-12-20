/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include "paddle/pten/core/allocator.h"
#include "paddle/pten/core/dense_tensor.h"
#include "paddle/pten/core/tensor_base.h"
#include "paddle/pten/core/tensor_meta.h"

namespace pten {

class CompatibleDenseTensorUtils;

/// \brief The SparseCsrTensor uses three 1-D DenseTensors to represent
/// the row index , column index and non zero elements of the original
/// DenseTensor.
/// where non_zero_crows_ represents the compressed row index,
/// non_zero_cols_ represents the column index of non zero elements in original
/// DenseTensor,
/// non_zero_elements_ represents the non zero elements of original DenseTensor.
class SparseCsrTensor : public TensorBase,
                        public TypeInfoTraits<TensorBase, SparseCsrTensor> {
 public:
  /// \brief Because sparse csr tensor is a kind of container, we give a default
  /// constructor to use for stl container. But the sparse csr tensor created
  /// with
  /// the default constructor is not practical.
  SparseCsrTensor() = default;

  /// \brief Because sparse csr tensor is a resource handle, we provide a
  /// default
  /// move constructor to support move semantics.
  SparseCsrTensor(SparseCsrTensor&& other) = default;

  /// \brief We do not recommend deep copy of sparse csr tensor because of its
  /// efficiency and complexity across devices. The operation is disabled here.
  SparseCsrTensor(const SparseCsrTensor& other) = delete;

  /// \brief Set the member tensor of the sparse csr tensor.
  /// \param non_zero_crows The compresessed row index of non zero elements in
  /// original dense tensor.
  /// \param non_zero_cols The column index of non zero elements in original
  /// dense tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  SparseCsrTensor(std::unique_ptr<DenseTensor> non_zero_crows,
                  std::unique_ptr<DenseTensor> non_zero_cols,
                  std::unique_ptr<DenseTensor> non_zero_elements,
                  const DDim& dims);

  /// \brief Destroy the tensor object and release exclusive resources.
  virtual ~SparseCsrTensor() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "SparseCsrTensor"; }

  /// \brief Returns the compressed row index of non zero elemetns in original
  /// dense tensor.
  /// \return The compressed row index of non zero elemetns in original dense
  /// tensor.
  const DenseTensor& non_zero_crows() const { return *non_zero_crows_; }

  /// \brief Returns the column index of non zero elemetns in original dense
  /// tensor.
  /// \return The column index of non zero elemetns in original dense tensor.
  const DenseTensor& non_zero_cols() const { return *non_zero_cols_; }

  /// \brief Returns the non zero elemetns in original dense tensor.
  /// \return The non zero elemetns in original dense tensor.
  const DenseTensor& non_zero_elements() const { return *non_zero_elements_; }

  /// \brief Return the number of non zero elements
  /// \return The number of non zero elements
  int64_t nnz() const;

  /// \brief Return the number of elements contained in original dense tensor
  /// \return The number of elements contained in original dense tensor
  int64_t numel() const { return product(dims_); }

  /// \brief Returns the dims of the original dense tensor.
  /// \return The dims of the original dense tensor.
  const DDim& dims() const noexcept override { return dims_; }

  /// \brief Returns the data type of the tensor.
  /// \return The data type of the tensor.
  DataType dtype() const noexcept override {
    return non_zero_elements_->dtype();
  }

  /// \brief Returns the data layout of the tensor.
  /// \return The data layout of the tensor.
  DataLayout layout() const { return DataLayout::SPARSE_CSR; }

  /// \brief Returns the data place of the tensor.
  /// \return The data place of the tensor.
  const Place& place() const override { return non_zero_elements_->place(); }

  /// \brief Test whether the non_zero_elements_ metadata is valid.
  /// \return Whether the non_zero_elements_ metadata is valid.
  bool valid() const noexcept { return non_zero_elements_->valid(); }

  /// \brief Test whether the non_zero_elements_ storage is allocated.
  /// return Whether the non_zero_elements_ storage is allocated.
  bool initialized() const override {
    return non_zero_elements_->initialized();
  }

  /// \brief Set the member tensor of the sparse csr tensor.
  /// \param non_zero_crows The compresessed row index of non zero elements in
  /// original dense tensor.
  /// \param non_zero_cols The column index of non zero elements in original
  /// dense tensor.
  /// \param non_zero_elements The non zero elements of original dense tensor.
  /// \param dims The dims of original dense tensor.
  void SetMemberTensor(std::unique_ptr<DenseTensor> non_zero_crows,
                       std::unique_ptr<DenseTensor> non_zero_cols,
                       std::unique_ptr<DenseTensor> non_zero_elements,
                       const DDim& dims);

  /// \brief resize sparse csr tensor.
  /// \param dims The dims of origin dense tensor.
  /// \param non_zero_num The number of non zero elements in origin dense tensor
  void Resize(const DDim& dims, const int64_t non_zero_num);

  /// \brief resize sparse csr tensor.
  /// \param dims The dims of origin dense tensor.
  /// \param non_zero_num The number of non zero elements in origin dense tensor
  void Resize(const std::shared_ptr<Allocator>& a,
              const DenseTensorMeta& meta,
              const int64_t non_zero_num);

  /// \brief Get a mutable data pointer of non_zero_crows.
  /// return a mutable data pointer of non_zero_crows.
  int64_t* mutable_non_zero_crows();

  /// \brief Get a mutable data pointer of non_zero_cols.
  /// return a mutable data pointer of non_zero_cols.
  int64_t* mutable_non_zero_cols();

  /// \brief Get a mutable data pointer of non_zero_elements.
  /// return a mutable data pointer of non_zero_elements.
  template <typename T>
  T* mutable_non_zero_elements();

 private:
  std::unique_ptr<DenseTensor> non_zero_crows_;
  std::unique_ptr<DenseTensor> non_zero_cols_;
  std::unique_ptr<DenseTensor> non_zero_elements_;
  DDim dims_;
};

}  // namespace pten
