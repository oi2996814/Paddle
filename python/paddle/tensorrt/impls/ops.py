# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved
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

from paddle.tensorrt.converter_utils import unary_op_converter
from paddle.tensorrt.register import converter_registry


@converter_registry.register("pd_op.sqrt", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sqrt_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.floor", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.exp", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.abs_", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sin", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cos", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.cosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.asinh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.acosh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.atanh", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.ceil", trt_version="trt_version_ge=8.0")
@converter_registry.register(
    "pd_op.reciprocal", trt_version="trt_version_ge=8.0"
)
@converter_registry.register("pd_op.erf", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.rsqrt", trt_version="trt_version_ge=8.0")
@converter_registry.register("pd_op.sign", trt_version="trt_version_ge=8.2")
@converter_registry.register("pd_op.round", trt_version="trt_version_ge=8.2")
def UnaryOpConverter(network, paddle_op, inputs):
    layer_output = unary_op_converter(network, paddle_op, inputs)
    return layer_output
