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
from tensorrt_test_base import TensorRTBaseTest

import paddle


class TestSqrtTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sqrt
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestFloorFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.floor
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestExpFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.exp
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestAbsFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.abs
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestAbsIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.abs
        self.api_args = {
            "x": np.random.randn(7, 3).astype("int64"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSinFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sin
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestCosFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cos
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestSinhFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sinh
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestCoshFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.cosh
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestAsinhFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.asinh
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestAcoshFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.acosh
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestCeilFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.ceil
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestRsqrtFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.rsqrt
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestReciprocalFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.reciprocal
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestErfFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.erf
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestSignFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sign
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestSignIntTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.sign
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


class TestRoundFloatTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.round
        self.api_args = {
            "x": np.random.randn(7, 3).astype("float32"),
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [3, 3]}
        self.opt_shape = {"x": [7, 3]}
        self.max_shape = {"x": [10, 3]}

    def test_trt_result(self):
        self.check_trt_result()

    def test_trt_result_fp16(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
