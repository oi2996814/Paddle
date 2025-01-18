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


def batch_norm_wrapper(x):
    batch_norm = paddle.nn.BatchNorm(num_channels=1, is_test=True)
    return batch_norm(x)


class TestBatchNormTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = batch_norm_wrapper
        self.api_args = {
            "x": np.arange(12).reshape([2, 1, 2, 3]).astype("float32")
        }
        self.program_config = {"feed_list": ["x"]}
        self.min_shape = {"x": [1, 1, 2, 3]}
        self.opt_shape = {"x": [2, 1, 2, 3]}
        self.max_shape = {"x": [5, 1, 2, 3]}

    def test_trt_result(self):
        self.check_trt_result()


def instance_norm_wrapper(x, weight, bias):
    return paddle.nn.functional.instance_norm(x, None, None, weight, bias)


class TestInstanceNormTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = instance_norm_wrapper
        self.api_args = {
            "x": np.arange(12).reshape([2, 2, 1, 3]).astype("float32"),
            "weight": np.random.random([2]).astype("float32"),
            "bias": np.random.random([2]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "weight", "bias"]}
        self.min_shape = {"x": [1, 2, 1, 3]}
        self.opt_shape = {"x": [2, 2, 1, 3]}
        self.max_shape = {"x": [5, 2, 1, 3]}

    def test_trt_result(self):
        self.check_trt_result()


class TestInstanceNormWith3DInputTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = instance_norm_wrapper
        self.api_args = {
            "x": np.arange(4).reshape([2, 2, 1]).astype("float32"),
            "weight": np.random.random([2]).astype("float32"),
            "bias": np.random.random([2]).astype("float32"),
        }
        self.program_config = {"feed_list": ["x", "weight", "bias"]}
        self.min_shape = {"x": [1, 2, 1]}
        self.opt_shape = {"x": [2, 2, 1]}
        self.max_shape = {"x": [5, 2, 1]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestInstanceNormWithNoneInputTRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = instance_norm_wrapper
        self.api_args = {
            "x": np.arange(12).reshape([2, 2, 1, 3]).astype("float32"),
            "weight": None,
            "bias": None,
        }
        self.program_config = {"feed_list": ["x", "weight", "bias"]}
        self.min_shape = {"x": [1, 2, 1, 3]}
        self.opt_shape = {"x": [2, 2, 1, 3]}
        self.max_shape = {"x": [5, 2, 1, 3]}

    def test_trt_result(self):
        self.check_marker(expected_result=False)


class TestGroupNormNCHWFP32TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.group_norm
        self.api_args = {
            "x": np.random.random([4, 32, 64, 64]).astype(np.float32),
            "num_groups": 2,
            "epsilon": 1e-05,
            "weight": np.random.randn(32).astype(np.float32),
            "bias": np.random.randn(32).astype(np.float32),
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x", "weight", "bias"]}
        self.min_shape = {"x": [1, 32, 64, 64]}
        self.opt_shape = {"x": [4, 32, 64, 64]}
        self.max_shape = {"x": [6, 32, 64, 64]}

    def test_trt_result(self):
        self.check_trt_result()


class TestGroupNormNCHWFP16TRTPattern(TensorRTBaseTest):
    def setUp(self):
        self.python_api = paddle.nn.functional.group_norm
        self.api_args = {
            "x": np.random.random([4, 32, 64, 64]).astype(np.float32),
            "num_groups": 2,
            "epsilon": 1e-05,
            "weight": np.random.randn(32).astype(np.float32),
            "bias": np.random.randn(32).astype(np.float32),
            "data_format": "NCHW",
        }
        self.program_config = {"feed_list": ["x", "weight", "bias"]}
        self.min_shape = {"x": [1, 32, 64, 64]}
        self.opt_shape = {"x": [4, 32, 64, 64]}
        self.max_shape = {"x": [6, 32, 64, 64]}

    def test_trt_result(self):
        self.check_trt_result(precision_mode="fp16")


if __name__ == '__main__':
    unittest.main()
