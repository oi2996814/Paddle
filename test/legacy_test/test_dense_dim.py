# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import unittest

import numpy as np

import paddle
from paddle.base import core


def coo_dense_dim_ref(coo, indices):
    return len(coo.shape) - indices.shape[0]


def csr_dense_dim_ref(crows, values):
    nze_dim = len(values.shape)
    batch_dim = len(crows.shape) - 1
    return nze_dim - batch_dim - 1


def dense_dense_dim_ref(dense):
    return len(dense.shape)


class TestDenseDimAPI(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]])
        coo_values = np.array([1.0, 2.0, 3.0, 4.0])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 3, 5])
        csr_cols = np.array([1, 3, 2, 0, 1])
        self.csr_values = np.array([1, 2, 3, 4, 5.0])
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tensor, csr_tensor, other_tensor]

    def test_dense_dim(self):
        expected_result = [
            coo_dense_dim_ref(self.tensors[0], self.coo_indices),
            csr_dense_dim_ref(self.csr_crows, self.csr_values),
            dense_dense_dim_ref(self.tensors[2]),
        ]
        places = [core.CPUPlace()]
        if core.is_compiled_with_cuda():
            places.append(core.CUDAPlace(0))

        for place in places:
            paddle.disable_static(place)
            for i, t in enumerate(self.tensors):
                self.assertEqual(t.dense_dim(), expected_result[i])


class TestDenseDimAPI1(TestDenseDimAPI):
    def setUp(self):
        self.dtype = "float64"
        self.coo_indices = np.array([[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]])
        coo_values = np.array([1.0, 2.0, 3.0, 4.0])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 3, 5])
        csr_cols = np.array([1, 3, 2, 0, 1])
        self.csr_values = np.array([1, 2, 3, 4, 5.0])
        csr_shape = [3, 4]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor([1, 2, 3, 4], dtype=self.dtype)
        self.tensors = [coo_tensor, csr_tensor, other_tensor]


class TestDenseDimAPI2(TestDenseDimAPI):
    def setUp(self):
        self.dtype = "int16"
        self.coo_indices = np.array([[0, 0, 1, 2], [0, 1, 1, 2], [0, 1, 1, 2]])
        coo_values = np.array([1.0, 2.0, 3.0, 4.0])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 4, 0, 2, 2, 0, 1, 2])
        csr_cols = np.array([0, 1, 0, 1, 0, 1, 1, 1])
        self.csr_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0])
        csr_shape = [3, 2, 2]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor([[1, 2, 3, 4]], dtype=self.dtype)
        self.tensors = [coo_tensor, csr_tensor, other_tensor]


class TestDenseDimAPI3(TestDenseDimAPI):
    def setUp(self):
        self.dtype = "int32"
        self.coo_indices = np.array([[0, 0, 1, 2], [0, 1, 1, 2]])
        coo_values = np.array([1.0, 2.0, 3.0, 4.0])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 4, 0, 2, 2, 0, 1, 2])
        csr_cols = np.array([0, 1, 0, 1, 0, 1, 1, 1])
        self.csr_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0])
        csr_shape = [3, 2, 2]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor(
            [[[1], [2], [3], [4]]], dtype=self.dtype
        )
        self.tensors = [coo_tensor, csr_tensor, other_tensor]


class TestDenseDimAPI4(TestDenseDimAPI):
    def setUp(self):
        self.dtype = "int64"
        self.coo_indices = np.array(
            [
                [0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 1, 2, 2],
                [0, 0, 1, 1, 0, 0, 0, 1],
                [0, 1, 0, 1, 0, 1, 1, 1],
            ]
        )
        coo_values = np.array([1, 2, 3, 4, 1, 2, 2, 4])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 4, 0, 2, 2, 0, 1, 2])
        csr_cols = np.array([0, 1, 0, 1, 0, 1, 1, 1])
        self.csr_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0])
        csr_shape = [3, 2, 2]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor(
            [[[[1, 2], [3, 4]], [[1, 2], [0, 0]], [[0, 2], [0, 4]]]],
            dtype=self.dtype,
        )
        self.tensors = [coo_tensor, csr_tensor, other_tensor]


class TestDenseDimAPI5(TestDenseDimAPI):
    def setUp(self):
        self.dtype = "uint8"
        self.coo_indices = np.array(
            [[0, 0, 0, 0, 0], [0, 0, 1, 2, 2], [0, 1, 0, 0, 1]]
        )
        coo_values = np.array([[1, 2], [3, 4], [1, 2], [0, 2], [0, 4]])
        coo_tensor = paddle.sparse.sparse_coo_tensor(
            self.coo_indices, coo_values, dtype=self.dtype
        )
        self.csr_crows = np.array([0, 2, 4, 0, 2, 2, 0, 1, 2])
        csr_cols = np.array([0, 1, 0, 1, 0, 1, 1, 1])
        self.csr_values = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 2.0, 4.0])
        csr_shape = [3, 2, 2]
        csr_tensor = paddle.sparse.sparse_csr_tensor(
            self.csr_crows,
            csr_cols,
            self.csr_values,
            csr_shape,
            dtype=self.dtype,
        )
        other_tensor = paddle.to_tensor(
            [[[[1, 2], [3, 4]], [[1, 2], [0, 0]], [[0, 2], [0, 4]]]],
            dtype=self.dtype,
        )
        self.tensors = [coo_tensor, csr_tensor, other_tensor]


class TestDenseDimAPIStatic(unittest.TestCase):
    def setUp(self):
        self.dtype = "float32"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.other_tensor_arr = np.array([[[1, 2, 3, 4]]]).astype(self.dtype)

    def test_is_coalesced(self):
        paddle.enable_static()
        with paddle.static.program_guard(
            paddle.static.Program(), paddle.static.Program()
        ):
            coo_indices = paddle.static.data(
                name='coo_indices',
                shape=self.coo_indices.shape,
                dtype='int64',
            )
            coo_values = paddle.static.data(
                name='coo_values',
                shape=self.coo_indices.shape,
                dtype=self.dtype,
            )
            coo = paddle.sparse.sparse_coo_tensor(
                coo_indices,
                coo_values,
                shape=self.coo_shape,
                dtype=self.dtype,
            )
            other = paddle.static.data(
                name='other',
                shape=self.other_tensor_arr.shape,
                dtype=self.dtype,
            )

            exe = paddle.static.Executor()
            exe.run(
                feed={
                    'coo_indices': self.coo_indices,
                    'coo_values': self.coo_values,
                    'other': self.other_tensor_arr,
                }
            )
            expected_result = [
                coo_dense_dim_ref(coo, self.coo_indices),
                dense_dense_dim_ref(self.other_tensor_arr),
            ]
            self.assertEqual(coo.dense_dim(), expected_result[0])
            self.assertEqual(other.dense_dim(), expected_result[1])
        paddle.disable_static()


class TestDenseDimAPIStatic1(TestDenseDimAPIStatic):
    def setUp(self):
        self.dtype = "float64"
        self.coo_indices = np.array(
            [[0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 1, 2]]
        ).astype('int64')
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3, 3]
        self.other_tensor_arr = np.array([[[[1, 2, 3, 4]]]]).astype(self.dtype)


class TestDenseDimAPIStatic2(TestDenseDimAPIStatic):
    def setUp(self):
        self.dtype = "int16"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 0, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.other_tensor_arr = np.array([[[1, 2, 3, 4]]]).astype(self.dtype)


class TestDenseDimAPIStatic3(TestDenseDimAPIStatic):
    def setUp(self):
        self.dtype = "int32"
        self.coo_indices = np.array(
            [[0, 1, 0, 1], [0, 0, 1, 2], [0, 0, 1, 2]]
        ).astype('int64')
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3, 3]
        self.other_tensor_arr = np.array([[1, 2, 3, 4]]).astype(self.dtype)


class TestDenseDimAPIStatic4(TestDenseDimAPIStatic):
    def setUp(self):
        self.dtype = "int64"
        self.coo_indices = np.array([[0, 0, 0, 1], [0, 2, 1, 2]]).astype(
            'int64'
        )
        self.coo_values = np.array([1.0, 2.0, 3.0, 4.0]).astype(self.dtype)
        self.coo_shape = [2, 3]
        self.other_tensor_arr = np.array([[1, 2, 3, 4]]).astype(self.dtype)


class TestDenseDimAPIStatic5(TestDenseDimAPIStatic):
    def setUp(self):
        self.dtype = "uint8"
        self.coo_indices = np.array([[0, 0, 1, 2, 2], [0, 1, 0, 0, 1]]).astype(
            'int64'
        )
        self.coo_values = np.array(
            [[1.0, 2.0], [3.0, 4.0], [1.0, 2.0], [0.0, 4.0], [2.0, 4.0]]
        ).astype(self.dtype)
        self.coo_shape = [3, 2, 2]
        self.other_tensor_arr = np.array([1, 2, 3, 4]).astype(self.dtype)


if __name__ == "__main__":
    unittest.main()
