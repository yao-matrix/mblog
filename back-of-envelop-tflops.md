# 如何估算模型训练 `T(FL)OPS efficiency`

## Naive 方法

以 `torchvision` 中的 `ResNet50-v1.5` 模型为例。

- **Step 1：** 获取模型的前向理论需求 `MACs（Multiply–ACcumulate）`
可使用 [thop](https://github.com/Lyken17/pytorch-OpCounter) 得到模型的前向 `MACS`。使用如下代码可得 `ResNet50-v1.5` 的前向 `MACs`为 `4.112G`。
    ```python
	from torchvision.models import resnet50
	from thop import profile, clever_format
	import torch
	model = resnet50()
	input = torch.randn(1, 3, 224, 224)
	macs, params = profile(model, inputs=(input,))
	print(clever_format([macs, params], "%.3f"))
	```
- **Step 2：** 估算模型在某个实测性能下每秒所需的 `T(FL)OPS`
估算公式以 OpenAI [AI and Compute](https://openai.com/blog/ai-and-compute/) 估算公式为基础：

    > `required_T(FL)OPS = (MACs_per_forward_pass) * (2 (FL)OPs / MAC) * 3 * (number_of_examples_per_second)`
    >
    > 这里 `3` 指的是 `1` 个前向加 `2` 个后向（data bp，weights bp）

    再由实测性能数据：

    > | accelerator | data type | bs | IPS |
    > |  ----  | ----  |  ----  | ----  |
    > | V100 | FP16 | 256 | 1325 |
    > | V100 | FP32 | 128 | 303.1 | 
    <!--  > | MLU290 M5 | FP16 | 256 | 892 |
    > | MLU290 M5 | FP32 | 256 | 572 | -->
    >
    > 以 V100 FP16 训练为例，有：
    >
    > `MACs per forward pass = 4.112G`
    >
    > `number of examples per second = 1325`
    >
    > `required_(FL)OPS = 4.112G * 2 * 3 * 1325 = 32.69T`
    >
    > 汇总结果为：
    >  | accelerator | data type | bs | IPS | required T(FL)OPS |
    > |  ----  | ----  |  ----  | ----  |  ----  | 
    > | V100 | FP16 | 256 | 1325 | 32.69 |
    > | V100 | FP32 | 128 | 303.1 | 7.478 | 
    <!-- > | MLU290 M5 | FP16 | 256 | 892 | 22.007 |
    > | MLU290 M5 | FP32 | 256 | 572 | 14.112 | -->

- **Step 3：** 估算模型理论峰值算力利用率
	- 理论峰值算力
	![Alt text](assets/back-of-envelop-tflops/image-0.png)
	- 理论峰值算力利用率
	  > peak_ratio = required_T(FL)OPS / peak_T(FL)OPS

	  > | accelerator | data type | bs | IPS | required TF(L)OPS | peak ratio |
	  > |  ----  | ----  |  ----  | ----  |  ----  |   ----  | 
	  > | V100 | FP16 | 256 | 1325 | 32.69 | 29.2% |
	  > | V100 | FP32 | 128 | 303.1 | 7.478 | 53% | 
	  <!-- | MLU290 M5 | FP16 | 256 | 892 | 22.007 | 34.4% |
	  | MLU290 M5 | FP32 | 256 | 572 | 14.112 | 22% | **MLU290 M5 peak ratio估算以INT31理论算力为base -->

## References
1. [NV Training Performance Benchmark](https://developer.nvidia.com/deep-learning-performance-training-inference)
2. [thop](https://github.com/Lyken17/pytorch-OpCounter)
3. [OpenAI AI and Compute](https://openai.com/blog/ai-and-compute/)

*写于 2021 年 12 月*
