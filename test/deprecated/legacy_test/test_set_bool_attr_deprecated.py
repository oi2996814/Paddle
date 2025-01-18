#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from paddle import base


class TestAttrSet(unittest.TestCase):
    def test_set_bool_attr(self):
        paddle.enable_static()
        x = paddle.static.data(
            name='x', shape=[-1, 3, 7, 3, 7], dtype='float32'
        )
        param_attr = base.ParamAttr(
            name='batch_norm_w',
            initializer=paddle.nn.initializer.Constant(value=1.0),
        )
        bias_attr = base.ParamAttr(
            name='batch_norm_b',
            initializer=paddle.nn.initializer.Constant(value=0.0),
        )
        bn = paddle.static.nn.batch_norm(
            input=x, param_attr=param_attr, bias_attr=bias_attr
        )
        block = base.default_main_program().desc.block(0)
        op = block.op(0)
        before_type = op.attr_type('is_test')
        op._set_attr('is_test', True)
        after_type = op.attr_type('is_test')
        self.assertEqual(before_type, after_type)


if __name__ == '__main__':
    paddle.enable_static()
    unittest.main()
