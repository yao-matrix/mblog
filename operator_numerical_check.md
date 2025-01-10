# 算子数值精度检查

## 基本公式

$\lvert{a - b}\rvert \leq atol + rtol * \lvert{b}\rvert$

其中：

- `atol`: **a**bsolute **tol**erance
- `rtol`: **r**elative **tol**erance 
- `NaN` 及 `Inf` 的行为：

    `NaN`s are treated as equal if they are in the same place and if `equal_nan=True`. `Inf`s are treated as equal if they are in the same place and of the same sign in both arrays.

## 主要软件库如何检查算子精度

### NumPy

Numpy 提供两个接口：

1. [numpy.allclose(a, b, rtol=1.e-5, atol=1.e-8, equal_nan=False)](https://github.com/numpy/numpy/blob/19989c21b6b7a45da80b21d3abde485542ae4eb1/numpy/core/numeric.py#L2189) 

2. [testing.assert_allclose(actual, desired, rtol=1e-07, atol=0, equal_nan=True)](https://numpy.org/doc/stable/reference/generated/numpy.testing.assert_allclose.html)

### PyTorch

PyTorch 采用 Numpy 相同的缺省阈值，API 为:

> [torch.allclose(input, other, rtol=1e-05, atol=1e-08, equal_nan=False)](https://pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose)

每个 `op` 可使用缺省阈值也可自设，具体每个 `op` 的阈值可见[此处代码](https://github.com/pytorch/pytorch/tree/master/test/); Caffe2 各算子的阈值可见[此处代码](https://github.com/pytorch/pytorch/tree/master/caffe2/python/operator_test)。

> **例子（[代码](https://github.com/pytorch/pytorch/blob/master/caffe2/python/operator_test/matmul_op_test.py)）：**
> 
> | op | 数据类型 | absolute tolerance | relative tolerance |
> | ----  | ---| ---| ---|
> | MatMul(FWD) | FP32 | 1e-4 | 1e-4 |
> | MatMul(FWD) | FP16 | 150 * 1e-4 | 150 * 1e-4 |

### TensorFlow

TF 使用如下 `test util` 判断：

> [assertAllCloseAccordingToType(a, b, rtol=1e-06, atol, float_rtol, float_atol, half_rtol, half_atol, bfloat16_rtol, bfloat16_atol, msg=None)](https://www.tensorflow.org/agents/api_docs/python/tf_agents/utils/test_utils/TestCase#assertAllCloseAccordingToType) 

还有一个简化版本，如下：

> [assertAllClose](https://www.tensorflow.org/agents/api_docs/python/tf_agents/utils/test_utils/TestCase#assertAllClose)


对不同的数据类型，缺省阈值如下：

|  数据类型   | absolute tolerance | relative tolerance |
|  ----  | ----  | ----  |
| FP64 | 2.22e-15 | 2.22e-15 |
| FP32 | 1e-6 | 1e-6 |
| FP16 | 1e-3 | 1e-3 |
| BF16 | 1e-2 | 1e-2 |

TF 缺省阈值的选择理由如下：

> [The default atol and rtol is `10 * eps`, where `eps` is the smallest representable positive number such that `1 + eps != 1`. This is about `1.2e-6` in 32bit, `2.22e-15` in 64bit, and `0.00977` in 16bit. See numpy.finfo.](https://www.tensorflow.org/api_docs/python/tf/compat/v1/assert_near)

具体每个 `op` 的阈值可见[相应 `op` 的 test code ](https://github.com/tensorflow/tensorflow/tree/fbca103a4b5137ca25f2d0f99ae883ffc3bac81b/tensorflow/python/ops)或[相应 kernel 的 test code](https://github.com/tensorflow/tensorflow/tree/fbca103a4b5137ca25f2d0f99ae883ffc3bac81b/tensorflow/python/kernel_tests)。

> **例子（[代码](https://github.com/tensorflow/tensorflow/blob/fbca103a4b5137ca25f2d0f99ae883ffc3bac81b/tensorflow/python/kernel_tests/math_ops/matmul_op_test.py)）：**
> | op |  数据类型   | absolute tolerance | relative tolerance |
> |  ----  | ----  | ----  | ----  |
> | MatMul(FWD) | FP32 | 3e-5 | 3e-5 |
> | MatMul(FWD)  | FP16 | 0.2 | 0.2 |


*写于 2022 年 2 月*