---
title: "将 LLM 微调至 1.58 比特：轻松 get 极致量化" 
thumbnail: /blog/assets/1_58_llm_extreme_quantization/thumbnail.png
authors:
- user: medmekk
- user: marcsun13
- user: lvwerra
- user: pcuenq
- user: osanseviero
- user: thomwolf
translators:
- user: MatrixYao
---

# 将 LLM 微调至 1.58 比特：轻松 get 极致量化

随着大语言模型（LLM）的规模和复杂度不断增长，降低其计算和能源成本的挑战也日益严峻。一个解决此问题的流行思路是量化，其将参数的精度从标准 16 位浮点（FP16）或 32 位浮点（FP32）降低到位宽更低的格式，如 8 比特或 4 比特。这种方法虽然显著减少了内存使用并加快了计算速度，但代价通常是牺牲准确性。过多降低精度可能会导致模型丢失关键信息，从而使效果变差。 

[BitNet](https://arxiv.org/abs/2402.17764) 是一种特殊的 transformer 架构，它仅用三个值表示每个参数：`(-1, 0, 1)`，因而提供每参数仅 1.58 比特（$log_2{3}$）的极致量化。唯一的问题是，它需要从头开始训练模型。虽然结果让人耳目一新，但并不是每个人都有预算来预训练 LLM。为了克服这个限制，我们探索了一些技巧，可以将现有模型微调到 1.58 比特！欲知详情，请看下文！

## 目录
- [长话短说](#长话短说)
- [深入了解 BitNet](#深入了解-BitNet)
- [1.58b 的预训练结果](#158b-的预训练结果)
- [1.58b 微调](#158b-微调)
- [自定义算子及基准测试](#自定义算子及基准测试)
- [总结](#总结)
- [致谢](#致谢)
- [更多资源](#更多资源)

## 长话短说

[BitNet](https://arxiv.org/abs/2402.17764) 是 Microsoft 研究院推出的一种模型架构，它使用极致量化，仅用三个值表示每个参数：-1、0 和 1。因此，训得的模型每参数仅占 1.58 比特，显著降低了计算和内存需求。 

在计算矩阵乘时，与 LLaMA LLM 使用了 FP16 加法和乘法运算不同，BitNet 只用了 INT8 加法。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/matmulfree.png" alt="The new computation paradigm of BitNet b1.58" style="width: 100%;"/>
  <figcaption>BitNet b1.58 的新计算范式（图源：BitNet 论文 https://arxiv.org/abs/2402.17764）</figcaption>
</figure>

理论上来讲，这会降低能耗。与 Llama 相比，BitNet b1.58 在矩阵乘算术运算上的能耗降低了 71.4 倍。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/energy_consumption.png" alt="Energy consumption of BitNet b1.58 compared to LLaMA" style="width: 100%;"/>
  <figcaption>BitNet b1.58 与 LLama 的能耗对比（图源：BitNet 论文 https://arxiv.org/abs/2402.17764）</figcaption>
</figure>

我们成功使用 BitNet 架构微调了 [Llama3 8B 模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)，并在下游任务上展现出了强大的性能。我们把生成的 8B 模型发布在 [HF1BitLLM](https://huggingface.co/HF1BitLLM) 下。一共有三个模型，其中两个模型是用不同的训练配置在 10B 词元上微调而得的，而第三个模型是在 100B 词元上微调而得的。值得注意的是，我们的模型在 MMLU 基准测试中超越了 Llama 1 7B 模型。

### 如何在 Transformers 中使用该模型

为了将 BitNet 架构集成到 Transformers 中，我们引入了一种称为 “bitnet” 的新量化方法（详见 [PR](https://github.com/huggingface/transformers/pull/33410)）。该方法主要用与 BitNet 架构兼容的专用 BitLinear 层替换标准 Linear 层，新层包括配套的激活动态量化、权重解包以及矩阵乘。 

在 Transformers 中加载和测试我们的模型非常简单，API 没有改动：

```python
model = AutoModelForCausalLM.from_pretrained(
    "HF1BitLLM/Llama3-8B-1.58-100B-tokens",
    device_map="cuda",
    torch_dtype=torch.bfloat16
)    
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

input_text = "Daniel went back to the the the garden. Mary travelled to the kitchen. Sandra journeyed to the kitchen. Sandra went to the hallway. John went to the bedroom. Mary went back to the garden. Where is Mary?\nAnswer:"

input_ids = tokenizer.encode(input_text, return_tensors="pt").cuda()
output = model.generate(input_ids, max_new_tokens=10)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```

仅需上述代码，其余一切交由幕后无缝处理，因此用户无需担心额外的复杂性，只需安装最新版本的 `transformers` 即可。 

如欲快速试一下模型，可参考[这个笔记本](https://colab.research.google.com/drive/1ovmQUOtnYIdvcBkwEE4MzVL1HKfFHdNT?usp=sharing)。

## 深入了解 BitNet

[BitNet](https://arxiv.org/abs/2402.17764) 用名为 BitLinear 的专用层替换了多头注意力和前馈网络中的传统线性层，该层的精度为三元（甚至在初始版本中为二元）。BitLinear 层使用三个值（为 -1、0 和 1）来量化权重，并将激活量化为 8 比特。BitLinear 层在推理和训练阶段的实现不相同，我们将在下一节详述。

三元精度训练的主要障碍是权重值是离散化的（经由 `round()` 函数实现），因此不可微。BitLinear 用一个很好的技巧解决了这个问题：[STE（Straight Through Estimator，直通估计器）](https://arxiv.org/abs/1903.05662)。STE 通过将其梯度近似为 1（即将 `round()` 视同恒等函数），从而允许梯度流经不可微的舍入操作。另一种解释方式是，STE 不是在舍入函数处停止梯度，而是让梯度通过，就好像舍入从未发生过一样，从而使得我们可以用标准的梯度优化技术来更新权重。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/bitlinear.png" alt="The architecture of BitNet with BitLinear layers" style="width: 100%"/>
  <figcaption>带有 BitLinear 层的 BitNet 架构（图源：BitNet 论文 https://arxiv.org/pdf/2310.11453）</figcaption>
</figure>

### 训练

我们以全精度进行训练，但使用张量级对称量化将权重量化为三元值。首先，我们计算权重矩阵绝对值的平均值，并将其作为缩放值。然后，我们将权重除以缩放值，并对结果进行四舍五入，最后将其值钳至 -1 到 1 之间，最后再反量化回原精度。

$$ scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|} $$

$$ W_q = \text{clamp}_{[-1,1]}(\text{round}(W*scale_w)) $$

$$ W_{dequantized} = W_q*scale_w $$

然后使用逐词元 absmax 量化将激活量化为指定的位宽（本例子中为 8 比特）（有关量化方法的全面介绍，请参阅[这个帖子](https://mlabonne.github.io/blog/posts/Introduction_to_Weight_Quantization.html)）。这一步涉及到将激活缩放到 8 比特位宽的范围 `[−128, 127]`。量化公式为：

$$ scale_x = \frac{127}{|X|_{\text{max}, \, \text{dim}=-1}} $$

$$ X_q = \text{clamp}_{[-128,127]}(\text{round}(X*scale_x)) $$

$$ X_{dequantized} = X_q * scale_x $$

为了使公式更清晰，下面分别给出了尺寸为 3x3 的权重和激活量化的示例：

---
<details>
  <summary>例 1：权重量化 </summary>

  令权重矩阵 $W$ 为：
  
  $$ W = 
  \begin{bmatrix}
  0.8 & -0.5 & 1.2 \\
  -1.5 & 0.4 & -0.9 \\
  1.3 & -0.7 & 0.2
  \end{bmatrix} $$

  **第 1 步：计算权重缩放值**
  
  使用公式：

  $$ scale_w = \frac{1}{\frac{1}{nm} \sum_{ij} |W_{ij}|} $$

  计算 $W$ 每个元素绝对值的平均值：

  $$ \frac{1}{nm} \sum_{ij} |W_{ij}| = \frac{1}{9}(0.8 + 0.5 + 1.2 + 1.5 + 0.4 + 0.9 + 1.3 + 0.7 + 0.2) = \frac{1}{9}(7.5) = 0.8333 $$

  因此，缩放值即为：

  $$ scale_w = \frac{1}{0.8333} \approx 1.2 $$

  **第 2 步：量化权重矩阵**
  
  使用公式：

  $$ W_q = \text{clamp}_{[-1, 1]}(\text{round}(W \times scale_w)) $$

  我们首先按比例 $scale_w \approx 1.2$ 缩放权重:
  $$ W \times scale_w = 
  \begin{bmatrix}
  0.8 \times 1.2 & -0.5 \times 1.2 & 1.2 \times 1.2 \\
  -1.5 \times 1.2 & 0.4 \times 1.2 & -0.9 \times 1.2 \\
  1.3 \times 1.2 & -0.7 \times 1.2 & 0.2 \times 1.2
  \end{bmatrix}
  =
  \begin{bmatrix}
  0.96 & -0.6 & 1.44 \\
  -1.8 & 0.48 & -1.08 \\
  1.56 & -0.84 & 0.24
  \end{bmatrix} $$

  然后，对矩阵的每个元素进行四舍五入并将它们钳在 $[-1, 1]$ 范围内:

  $$ W_q = 
  \begin{bmatrix}
  1 & -1 & 1 \\
  -1 & 0 & -1 \\
  1 & -1 & 0
  \end{bmatrix} $$

  **第 3 步：对权重进行反量化**
  
  最后，使用以下方法对权重进行反量化：

  $$ W_{dequantized} = W_q \times scale_w $$

  代入 $scale_w$，有：
  
  $$ W_{dequantized} = 
  \begin{bmatrix}
  1 \times 1.2 & -1 \times 1.2 & 1 \times 1.2 \\
  -1 \times 1.2 & 0 \times 1.2 & -1 \times 1.2 \\
  1 \times 1.2 & -1 \times 1.2 & 0 \times 1.2
  \end{bmatrix}
  =
  \begin{bmatrix}
  1.2 & -1.2 & 1.2 \\
  -1.2 & 0 & -1.2 \\
  1.2 & -1.2 & 0
  \end{bmatrix} $$

</details>

<details>
  <summary>例 2：激活矩阵量化</summary>

  令激活矩阵 $X$ 为：

  $$ X = 
  \begin{bmatrix}
  1.0 & -0.6 & 0.7 \\
  -0.9 & 0.4 & -1.2 \\
  0.8 & -0.5 & 0.3
  \end{bmatrix} $$

  **第 1 步：计算激活缩放值**  

  按行（或通道），计算最大绝对值：

  - **第 1 行**：最大绝对值 = 1.0
  - **第 2 行**：最大绝对值 = 1.2
  - **第 3 行**：最大绝对值 = 0.8

  计算每行的缩放因子：

  $$ \text{scale} = \begin{bmatrix}
  \frac{127}{1.0} \\
  \frac{127}{1.2} \\
  \frac{127}{0.8}
  \end{bmatrix}
  =
  \begin{bmatrix}
  127 \\
  105.83 \\
  158.75
  \end{bmatrix} $$


  **第 2 步：量化激活矩阵**  

  使用公式：
  
  $$ X_q = \text{clamp}_{[-128,127]}(\text{round}(X \times \text{scale})) $$

  对激活进行缩放：
  
  $$ X \times \text{scale} = 
  \begin{bmatrix}
  1.0 \times 127 & -0.6 \times 127 & 0.7 \times 127 \\
  -0.9 \times 105.83 & 0.4 \times 105.83 & -1.2 \times 105.83 \\
  0.8 \times 158.75 & -0.5 \times 158.75 & 0.3 \times 158.75
  \end{bmatrix}
  =
  \begin{bmatrix}
  127 & -76.2 & 88.9 \\
  -95.2 & 42.3 & -127 \\
  127 & -79.4 & 47.6
  \end{bmatrix} $$

  对矩阵的每个元素进行四舍五入，并将它们钳在 $[-128, 127]$ 范围内:

  $$ X_q = 
  \begin{bmatrix}
  127 & -76 & 89 \\
  -95 & 42 & -127 \\
  127 & -79 & 48
  \end{bmatrix} $$

  **第 3 步：对激活进行反量化**  

  最后，使用以下方法对激活进行反量化：
  
  $$ X_{dequantized} = X_q \times \frac{1}{\text{scale}} $$
  
  代入公式，有：
  
  $$ X_{dequantized} = 
  \begin{bmatrix}
  127 \times \frac{1}{127} & -76 \times \frac{1}{127} & 89 \times \frac{1}{127} \\
  -95 \times \frac{1}{105.83} & 42 \times \frac{1}{105.83} & -127 \times \frac{1}{105.83} \\
  127 \times \frac{1}{158.75} & -79 \times \frac{1}{158.75} & 48 \times \frac{1}{158.75}
  \end{bmatrix}
  =
  \begin{bmatrix}
  1.0 & -0.6 & 0.7 \\
  -0.9 & 0.4 & -1.2 \\
  0.8 & -0.5 & 0.3
  \end{bmatrix} $$

</details>

---

我们在量化激活之前使用了层归一化（LN）对其方差进行归一化：

$$ \text{LN}(x) = \frac{x - E(x)}{\sqrt{\text{Var}(x) + \epsilon}} $$

这里 ϵ 取值很小，其主要目的是防溢出。

如前所述，`round()` 函数是不可微的。我们使用 `detach()` 以在后向传播时实现可微 STE：

```python
# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch
import torch.nn as nn 
import torch.nn.functional as F

def activation_quant(x):
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127) / scale
    return y
 
def weight_quant(w):
    scale = 1.0 / w.abs().mean().clamp_(min=1e-5)
    u = (w * scale).round().clamp_(-1, 1) / scale
    return u

class BitLinear(nn.Linear):
    """
    Only for training
    """
    def forward(self, x):
        w = self.weight
        x_norm = LN(x)
        
        # A trick for implementing Straight−Through−Estimator (STE) using detach()
        x_quant = x_norm + (activation_quant(x_norm) - x_norm).detach()
        w_quant = w + (weight_quant(w) - w).detach()
        
        # Perform quantized linear transformation
        y = F.linear(x_quant, w_quant)
        return y
```

### 推理

推理时，我们只需将权重量化为三元值，无需反量化。使用相同的方法将激活处理成 8 比特，然后使用高效算子执行矩阵乘，最后除以权重和激活缩放因子。这么做可以显著提高推理速度，特别是当硬件有相应的加速模块的情况下。你可以看到训练期间的反量化实现有所不同，因为矩阵乘是在 fp16/bf16/fp32 精度下进行的，以保证训练过程的数值精度。

```python
# Adapted from https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf
import torch
import torch.nn as nn 
import torch.nn.functional as F

def activation_quant_inference(x):
    x = LN(x)
    scale = 127.0 / x.abs().max(dim=-1, keepdim=True).values.clamp_(min=1e-5)
    y = (x * scale).round().clamp_(-128, 127)
    return y, scale
 
class BitLinear(nn.Linear):
    """
    Only for training
    """
    def forward(self, x):
        w = self.weight # weights here are already quantized to (-1, 0, 1)    
        w_scale = self.w_scale  
        x_quant, x_scale = activation_quant_inference(x)
        y = efficient_kernel(x_quant, w) / w_scale / x_scale
        return y
```

## 1.58b 的预训练结果

在尝试微调之前，我们首先重现 BitNet 论文的预训练结果。我们从一个小数据集 [tinystories](https://huggingface.co/datasets/roneneldan/TinyStories) 和 [Llama3 8B 模型](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) 开始。正如论文所述，我们也发现添加 LN 层可以提高模型性能。举个例子，我们发现，同样经过 2000 步的训练，没有归一化的情况下验证集困惑度为 6.3，而在归一化的情况下为 5.9，两种情况下的训练都很稳定。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/pre-training.png" alt="Pre-training plots without (blue) & with (green) layer normalisation" style="width: 100%;"/>
  <figcaption>有（橙色）无（蓝色）LN 层两种情况下的预训练损失曲线</figcaption>
</figure>

虽然预训练看起来非常有意思，但只有少数机构能够负担得起这么高的设备成本。反之，现在已经有很多强大的预训练模型，如果我们可以将它们转换为 1.58 比特，那就有用多了。但有报告表明，微调效果不如预训练那么好，因此我们开始着手进行调查，看看 1.58b 微调是否可行。

## 1.58b 微调

我们尝试对预训练的 Llama3 8B 权重进行微调，发现模型的表现稍有提升，但不及预期。 

> **注意：** 所有实验均使用 [Nanotron](https://github.com/huggingface/nanotron) 进行。如果你有兴趣尝试 1.58 比特预训练或微调，可参阅此 [PR](https://github.com/huggingface/nanotron/pull/180)。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/finetuning_basic.png" alt="Fine-tuning plot compared to pre-training plot" style="width: 100%;"/>
  <figcaption>微调损失曲线与预训练损失曲线对比</figcaption>
</figure>

为了理解背后的原因，我们尝试检查随机初始化模型和预训练模型的权重分布，以期发现潜在问题。

<div style="display: flex; justify-content: center;">
  <figure style="margin-right: 20px; text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/poids_aléatoires.png" alt="Random weights distribution (2 merged stds)" style="width: 400px;" />
    <figcaption>随机权重分布（包含 2 个正态分布）</figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/poids_llama3.png" alt="Pre-trained Llama3 weights distribution" style="width: 400px;" />
    <figcaption>预训练 Llama3 权重分布</figcaption>
  </figure>
</div>

两组权重的缩放值分布分别为： 

<div style="display: flex; justify-content: center;">
  <figure style="margin-right: 20px; text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/scales_random.png" alt="Random weights scales distribution" style="width: 400px;" />
    <figcaption>随机权重缩放值分布</figcaption>
  </figure>
  <figure style="text-align: center;">
    <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/scales_llama3.png" alt="Pre-trained Llama3 weights distribution" style="width: 400px;" />
    <figcaption>预训练 Llama3 权重缩放值分布</figcaption>
  </figure>
</div>

随机初始化权重分布由两个正态分布混合而得：

- 一个标准偏差（std）为 $0.025$
- 另一个标准差为 $\frac{0.025}{\sqrt{2 \cdot \text{num\_hidden\_layers}}} = 0.00325$

这是由于 `nanotron` 中的列线性权重和行线性权重使用了不同的标准差。在量化时，两者权重矩阵的缩放值分别为 50.25 或 402，其计算公式为权重的平均绝对值的倒数：`scale = 1.0 / w.abs().mean().clamp_(min=1e-5)`

- 当 $\text{scale} = 50.25$ 时，有 $w.abs().mean() = 0.0199$，可得 $\text{std} = 0.025$，即第一个标准差。标准差推导公式基于 $|w|$ 的半正态分布的期望而得，如下：  
  $$\mathbb{E}(|w|) = \text{std}(w) \cdot \sqrt{\frac{2}{\pi}}$$
- 当 $\text{scale} = 402$ 时，有 $w.abs().mean() = 0.0025$，可得 $\text{std} = 0.00325$

另一方面，预训练权重的分布看起来像标准差为 0.013 的正态分布。

显然，预训练模型的初始信息更多（缩放值分布更连续），而随机初始化模型初始信息几乎为零，再随着时间的推移不断增加。我们的结论是，随机初始化模型从最少的初始信息开始，进而完成渐进的学习过程；而微调时，虽然预训练模型提供了不少先验信息，但 BitLinear 层的引入会使模型失去所有先验信息。

为了改善微调结果，我们尝试了不同的技术。例如，我们没有使用张量级量化，而是尝试按行和按列量化，以最大程度保留 Llama 3 权重中的信息。我们还尝试改变缩放值的计算方式：我们不再仅仅以权重的平均绝对值作为缩放值，而是以异常值的平均绝对值作为尺度（异常值数值超过 k * mean_absolute_value，k 为常数，实验中可配置），但我们并没有观察到很大的改进。

```python
def scale_outliers(tensor, threshold_factor=1):
    mean_absolute_value = torch.mean(torch.abs(tensor))
    threshold = threshold_factor * mean_absolute_value
    outliers = tensor[torch.abs(tensor) > threshold]
    mean_outlier_value = torch.mean(torch.abs(outliers))
    return mean_outlier_value

def weight_quant_scaling(w):
    scale = 1.0 / scale_outliers(w).clamp_(min=1e-5)
    quantized_weights = (w * scale).round().clamp_(-1, 1) / scale
    return quantized_weights
```
我们观察到，随机权重和 Llama 3 权重的初始损失大致相同，均为 13 左右。这表明 Llama 3 模型在引入量化时会丢失所有先验信息。为了进一步研究模型在此过程中丢失了多少信息，我们尝试了分组量化。

为确保代码功能没问题，我们首先将组大小设置为 1，也就是说没有量化。在这种情况下，损失从 1.45 开始，与我们正常微调看到的一样。然而，当我们将组大小增加到 2 时，损失跃升至 11 左右。这表明即使设置最小的大小为 2 的组，模型仍会丢失几乎全部信息。

为了解决这个问题，我们考虑了引入渐进式量化的可能性，不一下子量化所有权重和激活。为此，我们引入了一个 `lambda` 值来控制该过程：

```python
lambda_ = ?
x_quant = x + lambda_ * (activation_quant(x) - x).detach()
w_quant = w + lambda_ * (weight_quant(w) - w).detach()
```

当 `lambda` 设为 0 时，不会发生量化，而当 `lambda=1` 时，则完全量化。

我们最初测试了一组离散的 `lambda` 值，如 0.25->0.5->0.75->1。然而，这种方法并未显著改善结果，主要是因为 `lambda=0.25` 已经比较大了，初始损失仍然很高。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/lambda_0.25.png" alt="Fine-tuning plot with lambda = 0.25->0.5->0.75->1" style="width: 100%;"/>
  <figcaption>lambda = 0.25->0.5->0.75->1 时的微调损失曲线</figcaption>
</figure>

因此，我们决定尝试根据训练步数动态调整 `lambda` 值。

```python
lambda_ = training_step / total_training_steps
```

该动态 `lambda` 值可以带来更好的损失收敛，但是当 `lambda` 设置为 1 时，推理困惑度（perplexity，ppl）结果仍然差强人意。我们意识到这可能是因为模型在 `lambda=1` 上的训练步数不够。为了解决这个问题，我们继续调整 `lambda` 值以改进训练过程。

```python
lambda_ = min(2 * training_step / total_training_steps, 1)
```

使用此配置，经过 2000 步后，我们得到：  

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/lambda_training_step.png" alt="Fine-tuning plot with lambda = min(2*training_step/total_training_steps, 1)" style="width: 100%;"/>
  <figcaption>使用 lambda = min(2*training_step/total_training_steps, 1) 进行微调的损失曲线</figcaption>
</figure>

可以看到，我们的微调方法总体上展示出更好的收敛性。你可以看到损失曲线在 1000 步左右略有上扬，此时 `lambda` 开始接近 `1`（即完全量化）。然而，过了这一段，损失立即再次开始收敛，将困惑度进一步降低到 4 左右。

尽管效果不错，但当我们在 WikiText 数据集（而不是微调数据集 tinystories）上测试量化模型时，得到的困惑度很高。这表明在特定数据集上以低比特模式微调模型会导致其丢失许多一般知识。这是有可能的，因为权重的三元最小表征可能在一个数据集和另一个数据集之间存在显著差异。为了解决这个问题，我们扩展了训练过程，以包含更大的 [FineWeb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb) 数据集。我们维持 `lambda` 值为：

```python
lambda_ = min(training_step/1000, 1)
```
选择这个 `lambda` 值的原因是我们的经验表明其可作为渐进式量化的一个很好的起点。然后，我们在 FineWeb-edu 数据集上使用 1e-4 的学习率训练 5000 步。此次训练使用的总 batch size 为 200 万，总计 100 亿个词元。

找到正确的学习率和正确的衰减很需要点功夫，这似乎是模型性能的关键因素。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/fineweb-edu.png" alt="Fine-tuning plot with warmup quantization on Fineweb-edu" style="width: 100%;"/>
  <figcaption>在 Fineweb-edu 上进行渐进式量化微调的损失曲线</figcaption>
</figure>

经过 Fineweb-Edu 的微调过程后，WikiText 数据集上的困惑度达到了 12.2，考虑到我们只使用了 100 亿个词元，结果相当不错。如果将数据量纳入考量，其他评估指标也可算不错（参见下文结果部分）。

我们还尝试对 lambda 接近 1 时的损失急剧增长进行平滑。为此，我们设计了一个 lambda 调度器，其首先呈指数增长，然后在接近 1 时趋于平稳。

```python
def scheduler(step, total_steps, k):
    normalized_step = step / total_steps
    return 1 - (1 - normalized_step)**k
```

针对不同的 k 值，我们可以得到下图:

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp_scheduler.png" alt="Exponential scheduler for different k values" style="width: 100%;"/>
  <figcaption>不同 k 值下的指数调度器</figcaption>
</figure>

我们使用性能最佳的学习率 1e-4 进行了 4 次实验，测试 k 值分别为 [4, 6, 8, 10] 下的损失曲线。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp_scheduler_results.png" alt="Fine-tuning plots with exponential scheduler" style="width: 100%;"/>
  <figcaption>使用指数调度器微调的损失曲线</figcaption>
</figure>

平滑效果很好，没有出现像线性调度器那样的尖峰。然而，困惑度并不好，都在 15 左右，并且下游任务的性能也没有提升。

我们还注意到一开始的尖峰，模型很难恢复回来。当 lambda = 0 时，基本上没有量化，因此损失开始很低，约为 2。与线性调度器类似（如上面的蓝色图所示），在第一步之后，还存在一个尖峰。为了消除它，我们尝试了另一个调度器（S 型调度程序），它启动期爬坡会比较缓慢，中期急剧上升到 1，最后在接近 1 时趋于平稳。

```python
def sigmoid_scheduler(step, total_steps, k):
    # Sigmoid-like curve: slow start, fast middle, slow end
    normalized_step = step / total_steps
    return 1 / (1 + np.exp(-k * (normalized_step - 0.5)))
```

对不同的 k 值，我们有以下曲线： 

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/sig_scheduler.png" alt="Sigmoid scheduler for different k values" style="width: 100%;"/>
  <figcaption>不同 k 值下的 S 形调度器</figcaption>
</figure>

这次我们用 [15, 20, 25, 40, 100] 这 5 个 k 值做实验: 

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/sig_scheduler_exps.png" alt="Finetuning plots with sigmoid scheduler" style="width: 100%;"/>
  <figcaption>使用 S 形调度器进行微调的损失曲线</figcaption>
</figure>

lambda 的急剧增加导致第 500 步左右不稳定，且并未解决第一个尖峰问题。然而，当 $k = 100$ 时，我们观察到下游任务有所改善（参见下文结果部分），尽管困惑度仍维持在 13.5 左右。与线性调度器相比，看上去 S 形调度器并没有带来明显的性能提升。

此外，我们还使用随机权重和各种学习率从头开始尝试训练模型。以期将微调方法与传统预训练方法的效果进行横向比较。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/exp-randoms.png" alt="Different Pre-training plots with different learning rates" style="width: 100%;"/>
  <figcaption>不同学习率的预训练损失曲线</figcaption>
</figure>

随机权重预训练而得的模型中没有一个比微调的模型表现得更好。预训练模型达到的最佳困惑度是 26，不及我们微调而得的模型。

### 扩展到 100B 词元！

我们将实验规模扩大到 1000 亿个词元，看看是否可以与 Llama 3 8B 的性能相匹配。总运行比较长，我们采用了两段式微调：第一段使用线性调度器运行较短的步数，保存下性能最佳的 checkpoint；第二段基于该 checkpoint 继续微调 4 万 5 千步。我们尝试了不同的学习率，获得的模型虽然在某些指标上与 Llama 3 模型表现非常接近，但平均而言，仍然落后。

下图给出了训练期间各个 checkpoint 的评估指标趋势：

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_100B.png" alt="Metrics evaluations during the training for different lrs" style="width: 100%;"/>
  <figcaption>不同 lrs 在训练期间的各指标得分</figcaption>
</figure>

平均分如下： 

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metric_avg.png" alt="Average evaluation during the training for different lrs" style="width: 100%;"/>
  <figcaption>不同 lrs 在训练期间的指标平均分</figcaption>
</figure>

### 小模型实验

我们还对 SmolLM 等小模型进行了一些初步实验，使用渐进式量化，我们并没有在小模型上观察到与在大模型相当的改进。这表明渐进式量化的有效性可能与模型大小和复杂度密切相关。

举个例子，以下是 [SmolLM 135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) 模型的损失曲线，我们比较了渐进式量化和完全量化。有趣的是，曲线紧密对齐，并且产生的困惑度并无显著差别。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/smol_llm_exp.png" alt="SmolLM fine-tuning experiment with & without warmup quantization" style="width: 100%;"/>
  <figcaption>渐进式量化和完全量化的 SmolLM 微调实验</figcaption>
</figure>

### 结果对比

与基线方法相比，BitNet 表现出强大的性能，尤其是在较低比特情况下。根据该论文，BitNet 取得了与 8 比特模型相当的得分，但推理成本却显著降低。在 4 比特模型的情况下，仅量化权重的方法优于同时量化权重和激活的方法，因为激活更难量化。然而，使用 1.58 比特权重的 BitNet 同时超越了仅量化权重方法和权重+激活双量化方法。

下表列出了 Llama3 8B 经过 10B 词元微调后各种指标的结果。我们将这些结果与其他模型架构的结果进行比较，以提供全面的性能图景（所有评估均在 [Nanotron](https://github.com/huggingface/nanotron) 模型格式上使用 [Lighteval](https://github.com/huggingface/lighteval) 完成）。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_comparison_updated.png" alt="Metrics comparison with Llama models" style="width: 100%;"/>
  <figcaption>与 Llama 模型的指标比较：Linear 表示线性 lambda 调度器，Sigmoid 表示 S 形 lambda 调度器（在我们的例子中 k = 100）
</figcaption>
</figure>

在使用三元权重对 100 亿个词元进行微调后，该模型表现出了令人印象深刻的性能，与那些经过更广泛训练的模型相比毫不逊色。举个例子，它优于 Bitnet 7B 模型，该模型是在包含 1000 亿词元的更大数据集上训练而得的。此外，它的性能比 FBI LLM（完全二值化 LLM）更好，后者是在更庞大的 1.26 万亿词元上蒸馏出来的模型。这凸显了微调方法“花小钱，办大事”的优势。

对于 100B 词元实验，我们的最佳 checkpoint 表现如下： 

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/metrics_100B_table.png" alt="Metrics comparison with Llama models for the model trained on 100B tokens" style="width: 100%;"/>
  <figcaption>在 100B 词元上训练的模型与 Llama 模型的指标比较 </figcaption>
</figure>

要复现这些结果，你可以参阅此 [PR](https://github.com/huggingface/nanotron/pull/174) 以将模型转换为 nanotron 格式，然后对权重进行解包（见函数 [unpack_weights](https://gist.github.com/MekkCyber/78c1532e8767e8da0588b778faf61866)），最后使用 lighteval 进行评估。

请注意，即使模型已经基于指令微调模型微调过，应用本方法时，仍然需要使用指令数据集进行微调。你可将原指令微调模型视为基础模型。

## 自定义算子及基准测试

为了真正节省模型存储和内存，我们将 BitNet 低精度权重打包成 `int8` 张量（这使得参数字节数从 8B 降低到 2.8B！）。因此，在推理过程中，必须在执行矩阵乘之前解包这些权重。我们用 CUDA 和 Triton 实现了自定义算子，以处理矩阵乘之前的动态解包。对于矩阵乘本身，我们采用了缓存分块矩阵乘技术。为了完全掌握这种方法，我们首先回顾一些 CUDA 编程基础知识。

### GPU 基础概念：线程、线程块和共享内存

在深入研究缓存分块矩阵乘之前，了解一些 GPU 基础概念非常重要：

- **线程和线程块**：GPU 同时执行数千个线程。这些线程被分组为线程块（Block），每个线程块独立运行。这些线程块进一步组成了网格（Grid），其代表整个问题空间。举个例子，在矩阵乘中，每个线程可能负责计算输出矩阵的一个元素。

- **共享内存**：每个线程块都可以访问有限数量的共享内存，其比全局显存（GPU 的主存）快得多。但是，共享内存的大小有限，并且仅在线程块内的所有线程之间共享。高效使用共享内存是提高 GPU 程序性能的关键。

### 矩阵乘的挑战

在简单的 GPU 矩阵乘实现中，每个线程通过直接从全局显存读取必要的元素来计算输出矩阵的每个元素。然而，由于以下原因，这种方法效率很低：

- **显存带宽**：与 GPU 核心执行计算的速度相比，访问全局显存相对较慢。如果每个线程直接从全局显存读取矩阵元素，则访存时间可能成为瓶颈。

- **冗余数据访问**：在矩阵乘中，输入矩阵的许多元素被多次使用。如果每个线程独立地从全局内存中获取所需的数据，则相同的数据可能会多次加载到 GPU 中，从而导致效率低下。例如，如果一个线程用于计算输出矩阵中的一个元素，则负责计算位置 (i, j) 处的元素的线程将需要从显存中加载矩阵 A 的第 i 行和矩阵 B 的第 j 列。但是，其他线程（例如计算位置 (i+1, j) 处的元素的线程）无法重用此数据，必须从全局显存中重新加载 B 的第 j 列。

### 分块思想

分块（tiling）是一种用来解决上述挑战的技术，目前主要应用于 FlashAttention 中以提高算子效率。基本思想是将矩阵划分为更小的子矩阵，称为矩阵块，这些矩阵块可以放入 GPU 的共享内存中。放弃一次性计算整个输出矩阵，而是将计算分解为更小的部分，逐块进行处理。

在矩阵乘的上下文中，这意味着将矩阵 A 和 B 划分成块，将这些矩阵块加载到共享内存中，然后对这些较小的块执行乘法。这种方法允许线程间重用存储在高速共享内存中的数据，从而减少访问全局显存的次数。

工作原理如下：

- **将数据块加载进共享内存**：每个线程块协作地将矩阵 A 的块和矩阵 B 的相应块从全局显存加载到共享内存中。每个矩阵块仅加载一次，然后线程块中的线程会多次重复使用该矩阵块。

- **计算部分积**：将矩阵块加载进共享内存后，每个线程都会计算部分积。由于线程块中的所有线程都在共享内存中的相同矩阵块上工作，因此它们可以有效地重用数据，而无需额外的全局显存访问。

- **结果累加**：计算一个矩阵块的部分积后，线程将矩阵 A 和 B 中的下一个矩阵块加载到共享内存中并重复该过程。结果累积在寄存器（或局部内存）中，一旦处理完所有矩阵，将输出矩阵元素的最终值写回全局显存。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/illustration_tiling.png" alt="Tiled Matrix multiplication illustration" style="width: 100%;"/>
  <figcaption>分块矩阵乘图解（图源：https://cnugteren.github.io/tutorial/pages/page4.html）</figcaption>
</figure>

**实际考量**

在实现缓存分块矩阵乘时，需要考虑以下几个因素：

- **矩阵块大小**：权衡矩阵块的大小以平衡可放入共享内存的数据量以及全局显存访问次数。
- **访存合并**：合并全局显存的存取操作，这意味着相邻的线程需要访问相邻的内存位置。
- **使用率**：优化每个线程块的线程数以及网格中的线程块数以最大化使用率，这意味着 GPU 上要有尽可能多的活动 warp（一个 warp 包括 32 个线程）以隐藏内存延迟。

### Triton 算子

以下是基准测试使用的 Triton 算子： 

```python
@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn, 
        stride_cm, stride_cn,
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  
        GROUP_SIZE_M: tl.constexpr,
):

    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.int32)

    for i in range(4) : 
        b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)
        for j in range(0, tl.cdiv(K // 4, BLOCK_SIZE_K) ):
            k = i * tl.cdiv(K // 4, BLOCK_SIZE_K) + j 

            # BLOCK_SIZE_K must be a divisor of K / 4 
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0)
            b_uint8 = tl.load(b_ptrs, mask=offs_k[:, None] < K // 4 - j * BLOCK_SIZE_K, other=0)
            mask = 3<<(2*i)
            b = ((b_uint8 & mask) >> (2*i))

            # We accumulate the tiles along the K dimension.
            tensor_full = tl.full((1,), 1, dtype=tl.int8)

            accumulator += tl.dot(a, (b.to(tl.int8) - tensor_full), out_dtype=tl.int32)

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def matmul(a, b):
    assert a.shape[1] == b.shape[0] * 4, "Incompatible dimensions, the weight matrix need to be packed"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    _, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c
```

### 代码解读

1. **确定矩阵块位置**

首先确定每个线程块负责计算输出矩阵的哪个矩阵块：

- `pid` 是每个线程块的唯一标识符，经由 `tl.program_id(axis=0)` 获得。
- 网格被分为一组线程块（每块大小为 `GROUP_SIZE_M`），每组处理输出矩阵的一部分。
- `pid_m` 和 `pid_n` 分别是矩阵块在 M 和 N 维度上的坐标。
- 计算偏移量（`offs_am`、`offs_bn`、`offs_k`）以确定线程块中每个线程将处理矩阵 A 和 B 的哪些元素

2.  **加载并计算矩阵块**

在 K 维度上循环迭代每个 `BLOCK_SIZE_K` 矩阵块。对每个矩阵块：

- **加载矩阵块**：从全局显存加载矩阵 A 和 B 的相应矩阵块。
- **解包矩阵 B**：假设矩阵 B 的元素被打包进了 `int8` 值，这意味着每四个元素会被打包进同一个字节。解包循环为：
    - 从全局显存将打包的 `int8` 值加载进 `b_uint8` 变量。
    - 将每个值解包以获得用于计算的实际权重。
- **点积**：计算相应 A 和 B 矩阵块的点积，并将结果累加至 `accumulator`。 `accumulator` 中存储了输出矩阵 C 的矩阵块的部分结果。

3. **存储结果**

处理完 K 维度上的所有矩阵后，存储在 `accumulator` 中的最终结果将被转换为`float16` 并写回全局显存中矩阵 C 的相应矩阵块。写入过程使用掩码以防止内存越界。

关于代码的更详细的说明，请参阅此 [PR](https://github.com/linkedin/Liger-Kernel/pull/195/files)。

### 基准测试

对我们写的自定义算子以及 “`@torch.compile` 解包权重 + BF16 矩阵乘”进行基准测试，发现这两种方法性能大致相当。为了确保基准测试的准确性，我们运行了超过 2000 次 matmul 操作，并对最后 1000 次的运行时间进行平均，以消除与初始加载或编译相关的任何低效率问题。下图展示了基准测试结果。我们还测试了各种矩阵大小，x 轴表示对数尺度上的乘法次数，y 轴表示平均时间（以毫秒为单位）。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/without_bitblas.png" alt="Triton kernel compared to torch.compile" style="width: 100%;"/>
  <figcaption>Triton 算子与 torch.compile 性能对比</figcaption>
</figure>

我们还尝试了 BitBlas，这是一个混合精度矩阵运算的软件库。其允许以较低精度的格式（如 INT8、INT4 甚至 INT2）而不是传统的 FP32 或 FP16 格式进行计算，从而优化性能。 

基准测试结果很有前景，因为 BitBlas 在低精度方面优于我们写的自定义算子以及 Torch 的 `matmul` 函数，如下图所示。

<figure style="text-align: center;">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/1.58llm_extreme_quantization/with_bitblas.png" alt="Bitblas benchmark" style="width: 100%;"/>
  <figcaption>Bitblas 基准测试</figcaption>
</figure>

然而，在模型加载过程中，BitBlas 需要编译适合权重矩阵形状的算子并将其存储至本地数据库中，这会增加初始加载时间。 

## 总结

综上所述，随着 LLM 不断变大，通过量化减少其计算需求至关重要。本文探讨了使用三元权重的 1.58 比特量化方法。虽然 1.58 比特的预训练模型是资源密集型的，但我们已经证明，通过一些技巧，可以将现有模型微调到这个精度级别，从而在不牺牲准确性的情况下达成高效的性能。通过专门的算子优化推理速度，BitNet 为推动 LLM 更加实用和可扩展提供了新的可能性。

## 致谢

我们衷心感谢 Leandro von Werra、Thomas Wolf 和 Marc Sun 在整个项目中提供的宝贵帮助和见解。我们还要感谢 Omar Sanseviero 和 Pedro Cuenca 在完善这篇博文上所作的贡献，以帮助我们向人工智能社区清晰有效地传达我们的发现。

此外，我们还要感谢 GeneralAI 团队在 BitNet 项目上所做的开创性工作。他们的研究是我们工作的基础，我们特别感谢他们论文中提供的清晰准确的数据。

## 更多资源
1. H. Wang et al., *BitNet: Scaling 1-bit Transformers for Large Language Models*. [arxiv 论文](https://arxiv.org/pdf/2310.11453)
2. S. Ma et al., *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits*. [arxiv 论文](https://arxiv.org/pdf/2402.17764)
3. S. Ma et al., *The Era of 1-bit LLMs: Training Tips, Code and FAQ*. [链接](https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf)
4. RJ. Honicky, *Are All Large Language Models Really in 1.58 Bits?*. [博文](https://learning-exhaust.hashnode.dev/are-all-large-language-models-really-in-158-bits)
5. L. Mao, *CUDA Matrix Multiplication Optimization*. [博文](https://leimao.github.io/article/CUDA-Matrix-Multiplication-Optimization/)
6. *Tutorial: OpenCL SGEMM tuning for Kepler*. [链接](https://cnugteren.github.io/tutorial/pages/page4.html)
7. *CUDAMODE*. [github](https://github.com/cuda-mode), [youtube](https://www.youtube.com/channel/UCJgIbYl6C5no72a0NUAPcTA)
8. Wen-mei W. Hwu, David B. Kirk, Izzat El Hajj, *Programming Massively Parallel Processors : A Hands-on Approach*

> 英文原文: <url> https://huggingface.co/blog/1_58_llm_extreme_quantization </url>
> 原文作者：Mohamed Mekkouri，Marc Sun，Leandro von Werra，Pedro Cuenca，Omar Sanseviero，Thomas Wolf
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
