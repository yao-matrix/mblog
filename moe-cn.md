---
title: "详解 MoE"
thumbnail: /blog/assets/moe/thumbnail.png
authors:
- user: osanseviero
- user: lewtun
- user: philschmid
- user: smangrul
- user: ybelkada
- user: pcuenq
translators:
- user: MatrixYao
---

# 详解 MoE

随着 Mixtral 8x7B 的发布（[公告](https://mistral.ai/news/mixtral-of-experts/)，[模型卡](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)），MoE transformer（Mixture of Experts，混合专家）模型已经成为开放 AI 社区的热门话题。本文，我们主要讨论 MoE 模型的基础模块、训练方式以及针对推理场景的主要考量。

我们开始吧！

## 目录

- [详解 MoE](#详解-moe)
  - [目录](#目录)
  - [太长不看版](#太长不看版)
  - [MoE 模型到底是什么？](#moe-模型到底是什么)
  - [MoE 简史](#moe-简史)
  - [什么是稀疏性？](#什么是稀疏性)
  - [MoE 的词元级负载均衡](#moe-的词元级负载均衡)
  - [MoE 与 transformer 模型](#moe-与-transformer-模型)
  - [Switch Transformer 模型](#switch-transformer-模型)
  - [使用路由 z-loss 提高训练稳定性](#使用路由-z-loss-提高训练稳定性)
  - [专家到底学了啥？](#专家到底学了啥)
  - [专家数量的增减对预训练有何影响？](#专家数量的增减对预训练有何影响)
  - [微调 MoE 模型](#微调-moe-模型)
  - [何时使用稀疏 MoE，何时使用稠密模型？](#何时使用稀疏-moe何时使用稠密模型)
  - [让 MoE 性能飞起](#让-moe-性能飞起)
    - [并行](#并行)
    - [容量因子和通信开销](#容量因子和通信开销)
    - [推理部署技术](#推理部署技术)
    - [高效训练](#高效训练)
  - [开源 MoE 项目](#开源-moe-项目)
  - [后续方向](#后续方向)
  - [资源](#资源)
  - [本文引用格式](#本文引用格式)


## 太长不看版

MoE 模型：
- 与稠密模型相比，**预训练速度更快**
- 与等参数量的稠密模型相比，推理速度**更快**
- 显存要求**高**，因为所有专家模型都需加载至显存
- 微调**相对麻烦**，但最近[有工作](https://arxiv.org/pdf/2305.14705.pdf)表明 MoE 指令微调**还是有希望的**

我们细细道来！

## MoE 模型到底是什么？

众所周知，提高模型质量的最重要的手段之一就是扩大模型的规模。在给定的算力预算下，用更少的步数训练出的大模型比用更多的步数训练出的小模型质量更好。

MoE 支持我们用更少的算力来预训练大模型，这意味着给定算力预算，我们可以训练础一个比相应的稠密模型大得多的稀疏模型，或者在更大的数据集上进行训练。而且，MoE 模型的预训练收敛速度更快，也就是说其可以用更少的步数达到与其对应的稠密模型相同的质量。

那么，MoE 到底是什么？作为 transformer 模型一个分支，MoE 对 transformer 模型作了两处主要改进：

- 用**稀疏 MoE 层**代替稠密前馈网络（feed-forward network，FFN）。MoE 层有一定数量的 “专家”（如 8 个），其中每个专家都是一个神经网络。当前，专家主要是 FFN，但其也可以是更复杂的网络，甚至可以是 MoE，此时会形成层次化的 MoE！

- **门控网络或路由**，决定将哪些词元发送给哪个专家。例如，在下图中，词元 “More” 被发送给第二个专家，词元 “Parameters” 被发送到第一个专家。稍后我们会讲到，也可以将某个词元同时发送给多个专家。如何将词元路由给专家是 MoE 模型的重要议题之一 - 路由器的参数是训练而得的，其与网络的其余部分是同时参与预训练的。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/00_switch_transformer.png" alt="Switch 层">
  <figcaption>[Switch Transformers](https://arxiv.org/abs/2101.03961)论文中的 MoE 层</figcaption>
</figure>

总结一下，在 MoE 中，我们将 transformer 模型的某些 FFN 层替换为 MoE 层，其中 MoE 层由门控网络和一定数量的专家组成。

尽管与稠密模型相比，MoE 的预训练更高效且推理速度更快，但天下没有免费的午餐：

- **训练：** 虽然 MoE 可以显著提高预训练的计算效率，但一直以来，我们受困于其微调问题，对 MoE 进行微调往往难以泛化并很容易过拟合。

- **推理：** 虽然 MoE 参数量很大，但在推理时，我们只会激活其中部分参数。因此，与相同参数量的稠密模型相比，其推理速度更快。然而，无论专家会不会被用到，都需要将其参数加载到内存中，这加大了对内存容量的要求。以 Mixtral 8x7B 为例，我们需要足够的内存来保存总共 47B 的参数。至于为什么是 47B 而不是 8 x 7B = 56B？这是因为在 MoE 模型中，只有专家 FFN 层需要额外的内存，其余模型参数是共享的。同时，假设每个词元仅激活两名专家，其推理速度 (FLOP) 与 12B 稠密模型（而不是 14B 模型）相当，因为虽然看上去有 2x7B 矩阵乘法的计算量，但其中有些层的计算是专家间共享的（我们将在下文详细阐述）。

现在我们已经大致了解了什么是 MoE，下面，我们简要回溯一下它的历史吧！

## MoE 简史

MoE 方法起源于 1991 年的论文 [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)。与集成方法类似，其主要想法是设计一个由各独立网络组成的系统，并对其进行有监督训练，其中每个网络负责处理训练集的不同子集。不同的网络（或专家）负责对输入空间的不同区域进行建模。那么，如何选择专家呢？其使用一个门控网络来确定每个专家的权重。在训练过程中，专家和门控网络会同时训练。

2010 年至 2015 年间，有两个不同的研究领域为 MoE 的进一步发展作出了贡献：

- **专家作为组件**：在传统的 MoE 设计中，整个系统由门控网络和多个专家组成。作为整体模型的 MoE 已经在 SVM、高斯过程及其他方法中获得了广泛的探索和使用。[Eigen、Ranzato 以及 Ilya](https://arxiv.org/abs/1312.4314) 的工作探索了把 MoE 作为某个深度神经网络的组件这一做法。该做法使得 MoE 能够作为多层网络中的一层而存在，从而使得模型可以又大又高效。

- **条件计算**：传统网络的每一层会处理所有输入数据。而 Yoshua Bengio 研究了根据输入词元动态激活或停用某些组件的方法。

这些工作激发了 NLP 社区对 MoE 方法的探索。具体来说，[Shazeer 等人](https://arxiv.org/abs/1701.06538)（包括 Geoffrey Hinton 、[Jeff Dean](https://www.informatika.bg/jeffdean) 以及 Google 的 Chuck Norris）通过引入稀疏性，将这个想法扩展到 137B 的 LSTM 上（LSTM 是当时最先进的 NLP 架构，由 Schmidhuber 提出），从而实现了大模型的快速推理。该工作主要应用于翻译场景，在工程方面也遇到了通信成本高、训练不稳定等诸多挑战。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/01_moe_layer.png" alt="LSTM 中的 MoE 层">
  <figcaption>Outrageously Large Neural Network 一文中 MoE 层</figcaption>
</figure>

MoE 允许训练数万亿参数的模型，如开源的 1.6T 参数的 Switch Transformers 等。计算机视觉领域对 MoE 也进行了相关探索，但本文重点关注 NLP 领域。

## 什么是稀疏性？

稀疏性利用了条件计算的思想。在稠密模型中，所有参数都会参与对输入的计算，但稀疏性使得我们可以仅激活系统的某些部分来参与对输入的计算。

我们来深入了解下 Shazeer 对 MoE 翻译的探索。条件计算的想法（每个样本仅激活网络的一部分）允许在不增加计算量的情况下扩大模型的规模，这使得每个 MoE 层甚至可以包含数千个专家网络。

这种设计带来了一些挑战。例如，虽然通常来讲，大 batch size 推理性能更好，但是由于各样本在 MoE 层激活的专家不同，MoE 层中每个专家的实际 batch size 会减少。举个例子，假设当前 batch 有 10 个词元，**其中 5 个词元被路由到了某个专家网络，而其他 5 个词元被路由到了其它 5 个不同的专家网络，这就会导致各专家网络获得的 batch size 不均匀以及算力利用率不足的问题**。我们会在下文的[让 MoE 性能飞起](#让-moe-性能飞起)一节讨论其它挑战及相应的解决方案。

我们该如何解决这个问题呢？向哪些专家（E）发送哪部分输入是由训练后的门控网络（G）决定的：

$$
y = \sum_{i=1}^{n} G(x)_i E_i(x)
$$

在上式中，所有专家都会作用于每个输入 - 其形式为加权乘法。但是，如果 G 为 0 会如何？此时，无需对相应的专家进行计算，因此可以节省计算量。典型的门控函数是什么？最传统的是使用 softmax 算子。经训练后，模型将据此计算应向哪些专家发送数据。

$$
G_\sigma(x) = \text{Softmax}(x \cdot W_g)
$$

Shazeer 的工作还探索了其他门控机制，如有噪 Top-K 门控。这种门控方法引入了一些（可调）噪声，并保留最高的 k 个值参与最终的 softmax 计算。如下：

1. 添加噪声
    
$$
H(x)_i = (x \cdot W_{\text{g}})_i + \text{StandardNormal()} \cdot \text{Softplus}((x \cdot W_{\text{noise}})_i)
$$
    
2. 选择 top k

$$
\text{KeepTopK}(v, k)_i = \begin{cases}
v_i & \text{if } v_i \text{ is in the top } k \text{ elements of } v, \\
-\infty & \text{otherwise.}
\end{cases}
$$

3. 计算 softmax

$$
G(x) = \text{Softmax}(\text{KeepTopK}(H(x), k))
$$

这种稀疏性引入了一些有趣的特性。如果 k 足够小（例如 2），相比于激活许多专家，训练和推理速度可以更快。但为什么不直接选择 top-1 专家呢？初始动机是为了让门控网络学到如何路由到不同专家的能力，因此训练时需要将每个输入至少路由至两个专家才行。[Switch Transformer 模型](#switch-transformer-模型) 一节重新审视了这一做法。

为什么要添加噪音呢？主要是为了负载均衡！

## MoE 的词元级负载均衡

如上文所讨论的，如果我们把所有的词元都发送给少数几个头部专家，训练效率将会变得低下。在正常的 MoE 训练中，门控网络会收敛至仅选择少数那几个专家。这种情况会不断自强化，因为受青睐的专家训练得更快，因此选择它们的频率也会更高。为了缓解这种情况，我们添加了**辅助损失**，以鼓励给予所有专家同等的重视。这种损失确保所有专家能收到数量大致相等的训练样本。下文我们将探讨专家容量的概念，其引入了一个阈值以度量每个专家被允许处理的最大词元数。在 `transformers` 中，用户可通过 `aux_loss` 来设置辅助损失。

## MoE 与 transformer 模型

对 transformer 模型而言，**扩大参数量可以提高性能**已是共识。因此，Google 在 [GShard](https://arxiv.org/abs/2006.16668) 一文也理所当然地选择了 transformer 模型，研究探索将其扩展到 600 亿以上参数。

GShard 将编码器和解码器中的所有 FFN 层，每隔一个将其替换为基于 top-2 门控的 MoE 层。下图展示了编码器部分的改动。这种设置对于大规模计算非常有利：当我们扩展到多个设备时，MoE 层可以分散至各个设备，而其他非 MoE 层需要每个设备复制一份。我们将在[让 MoE 性能飞起](##让-moe-性能飞起) 一节对此进行进一步讨论。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/02_moe_block.png" alt="MoE transformer 编码器">
  <figcaption>GShard 论文中的 MoE transformer 编码器</figcaption>
</figure>

为了在大模型上保持负载均衡及效率，除了上一节讨论的辅助损失之外，GShard 作者还引入了一些改动：

- **随机路由（random routing）**：在 top-2 路由时，除了必选 top-1 专家外，我们会随机选择第 2 个专家，该专家的被选概率与其权重成正比。

- **专家容量（expert capacity）**：我们可以设置一个阈值，用于规定每个专家最多可以处理的词元数。如果两个专家都满了，则该词元被视为溢出，可以将其通过残差连接直接发送给下一层（也可以直接丢弃）。这个概念是 MoE 最重要的概念之一。为什么要引入专家容量这个概念？虽然所有张量的形状都是在编译时静态确定的，但我们无法提前知道每个专家在运行时获得的词元数，因此我们需要容量因子来帮助我们进行运行时控制。

GShard 论文为确定适合 MoE 的并行计算模式作出了贡献，但对此的讨论超出了本文的范畴。

**注意：**在推理时，只有部分专家会被激活。同时，还有很多计算是共享的，如自注意力机制就是应用于所有词元的。这就是为什么当我们讨论含有 8 个专家的 47B 模型时，我们认为其计算量相当于一个 12B 的稠密模型。如果使用 top-2 门控的话，其参与计算的参数大约有 14B，但由于注意力等操作是共享的，实际计算量相当于 12B 的稠密模型。

## Switch Transformer 模型

尽管 MoE 表现出很大的潜力，但其仍面临着训练和微调不稳定的问题。[Switch Transformers](https://arxiv.org/abs/2101.03961) 论文对这些课题进行了深入的探讨，因此意义重大。作者甚至在 Hugging Face 上发布了 [1.6 万亿参数的 MoE 模型](https://huggingface.co/google/switch-c-2048)，其专家数为 2048，你可以使用 `transformers` 来运行它。Switch Transformers 的预训练速度比 T5-XXL 快 4 倍。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/03_switch_layer.png" alt="Switch Transformer 层">
  <figcaption>Switch Transformers 一文中的 Switch Transformer 层</figcaption>
</figure>

和 GShard 一样，作者用 MoE 层替换了 FFN 层。Switch Transformers 论文引入了 Switch Transformer 层，该层同时接收两个输入（两个不同的词元）且每个输入有四个备选专家。

与使用至少两名专家的最初想法相反，Switch Transformer 使用简化的单专家策略。这种方法的好处有：

- 减少了路由计算
- 每个专家的 batch size 至少可以减半
- 通信成本降低
- 保证质量

Switch Transformers 一文也探讨了专家容量的概念。

$$
\text{Expert Capacity} = \left(\frac{\text{tokens per batch}}{\text{number of experts}}\right) \times \text{capacity factor}
$$

上式中容量定义为 batch 中的总词元数除以专家数。容量因子大于 1 是我们为词元不完全均衡的情形提供的一个缓冲，但同时，增加容量也会导致设备间通信更加昂贵，因此需要权衡。论文表明，Switch Transformer 在容量因子比较小（1-1.25）时表现较好。

Switch Transformer 作者还重新审视并简化了上文提及的负载均衡损耗。在训练期间，对于每个 Switch 层，把辅助损失加到模型总损失中。该损失鼓励均匀化路由，并可使用超参进行设定各专家在辅助损失中的权重。

作者还尝试了选择性混合精度，例如使用 `bfloat16` 训练专家，同时使用全精度计算其他部分。较低的精度可以降低进程间的通信成本、计算成本以及用于存储张量的内存成本。最初，作者试验了对专家和门控网络都使用 `bfloat16` 进行训练，但训练结果更加不稳定。一个重要原因是路由计算：由于路由时会使用幂函数，因此更高的精度非常重要。为了减轻数值不稳定性，路由计算也需采用全精度。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/04_switch_table.png" alt="表明选择性精度可以不损失精度的表格">
  <figcaption>使用选择性精度可以在不损失质量的前提下加速收敛</figcaption>
</figure>

这里有个为摘要任务微调 Switch Transformer 模型的[笔记本](https://colab.research.google.com/drive/1aGGVHZmtKmcNBbAwa9hbu58DDpIuB5O4?usp=sharing)供参考，但我们建议先阅读本文的[微调 MoE 模型](#微调-moe-模型)一节。

Switch Transformer 的模型架构为编码器-解码器，其实现了 T5 模型的 MoE 版。而 [GLaM](https://arxiv.org/abs/2112.06905) 论文则意在进一步扩大模型的规模，其仅用 1/3 的能源就可以训练出一个与 GPT-3 质量相当的模型（这主要归功于训练 MoE 所需的计算量较少，其可以将碳足迹减少一个数量级）。作者的关注点主要在解码器模型以及少样本和零样本评估，而不在微调。论文使用了 top-2 路由以及更大的容量因子。此外，论文还探索了根据训练和评估过程中的算力预算机动调整容量因子的做法。

## 使用路由 z-loss 提高训练稳定性

前面讨论的负载均衡损失可能会导致数值不稳定。有不少提高稀疏模型训练的数值稳定性的方法，但其中很多会牺牲质量。例如，引入 dropout 可以提高稳定性，但会导致模型质量下降；而添加更多乘法算子可以提高质量，但会降低稳定性。

[ST-MoE](https://arxiv.org/abs/2202.08906) 论文引入了路由 z-loss，其通过惩罚进入门控网络的大 logits 显著提高了训练稳定性，同时并不会降低质量。由于这种损失会鼓励减小各张量的模，从而有助于降低舍入误差，这对门控函数这类指数函数很有帮助。建议读者阅读该论文以了解详情。

## 专家到底学了啥？

ST-MoE 作者观察到编码器的专家网络各自习得了某一组词元或浅层概念的专门知识，如，有标点符号专家、专有名词专家等等。另一方面，解码器的专家网络的特化程度则较低。作者还在多语种数据集上进行了训练。尽管大家可能会期待每个专家网络会各自习得某一种语言的专门知识，但事实恰恰相反：由于词元路由和负载均衡的作用，没有一个专家网络专门习得了某一特定语言的知识。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/05_experts_learning.png" alt="专家网络习得了特定词元组的知识">
  <figcaption>ST-MoE 论文中的表格，其展示了哪些词元组被发送给了哪个专家。
</figcaption>
</figure>

## 专家数量的增减对预训练有何影响？

更多的专家可以提高样本效率和收敛速度，但收益是递减的（尤其当专家数大于 256 或 512 后），同时也会要求更大的推理内存。在大 Switch Transformer 模型上得到的结论对小模型同样适用，即使是那种每层仅有 2、4 或 8 个专家的小模型。

## 微调 MoE 模型

> 4.36.0 以上版本的 `transformers` 增加了对 Mixtral 的支持。你可以使用 `pip install "transformers==4.36.0 --upgrade` 进行安装。

稠密模型和稀疏模型之间的过拟合情况有很大不同。稀疏模型更容易过拟合，因此我们可以在专家网络内使用更高的正则化（例如 dropout）（例如，我们可以为稠密层设一个 dropout 率，为稀疏层另设一个更高的 dropout 率）。

另一个问题是*微调时是否要使用辅助损失*。ST-MoE 作者尝试去掉辅助损失，即使丢弃了高达 11% 的词元，也没有明显影响质量。看上去，词元级 dropout 是一种有助于防止过拟合的正则化方法。

Switch Transformers 论文观察到，在预训练困惑度相同的情况下，稀疏模型在下游任务上的表现比稠密模型更差，尤其是在 SuperGLUE 等推理密集型的任务上。但同时，对于 TriviaQA 等知识密集型任务，稀疏模型的表现却异常出色。作者还观察到，在微调时减少激活专家数有助于提高模型的表现[译者注：指在微调时设置expert dropout]。另外，稀疏模型在较小的任务上表现较差，而在较大的任务上表现良好，这一观察进一步证实了其存在一定的泛化问题。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/06_superglue_curves.png" alt="微调的学习曲线">
  <figcaption>在小任务（左）上，我们可以看到明显的过拟合，因为稀疏模型在验证集上的表现要差得多。在较大的任务（右）上，MoE 表现良好。该图来自 ST-MoE 论文。</figcaption>
</figure>

我们尝试冻结所有非专家部分的权重，发现效果出现了巨大的下降，这不奇怪，因为 MoE 层占了模型参数量的大部分[译者注：这里的意思是说，MoE 层参数量比较大所以容易微调不充分]。于是，我们尝试相反的方法：冻结所有 MoE 层的参数，而微调非 MoE 部分，事实证明其效果与全模型微调相差无几。这种做法有助于加速微调过程并减少微调所需的内存。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/07_superglue_bars.png" alt="仅更新非 MoE 层的微调效果不错">
  <figcaption>通过冻结 MoE 层，我们可以在保持质量的同时加快训练速度。该图来自 ST-MoE 论文。</figcaption>
</figure>

微调稀疏 MoE 时的最后一个要注意的点是其微调超参配置与稠密模型不同 - 如，稀疏模型在微调时往往偏好较小的 batch size 及较高的学习率。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/08_superglue_dense_vs_sparse.png" alt="稀疏模型和稠密模型的质量随着 batch size 和学习率的变化趋势示意图">
  <figcaption>稀疏模型的质量随着学习率的增加及 batch size 的减小而提高。该图来自 ST-MoE 论文。</figcaption>
</figure>

至此，你可能会对 MoE 微调有点悲观。但最新的论文 [MoEs Meets Instruction Tuning](https://arxiv.org/pdf/2305.14705.pdf)（2023 年 7 月）似乎带来了一些好消息，该论文做了以下几个实验：

- 单任务微调
- 多任务指令微调
- 先做多任务指令微调，再做单任务微调

当作者对 MoE 模型及其对应的 T5 模型进行微调时，对应的 T5 模型效果更好。当作者对 Flan T5（T5 的指令微调版模型）MoE 进行微调时，MoE 的表现明显更好。不仅如此，Flan-MoE 相对于 MoE 的改进大于 Flan T5 相对于 T5 的改进，这表明 MoE 可能比稠密模型更能受益于指令微调。且任务数越多 MoE 模型的受益度越高。与前作的结论不同，本论文的实验表明辅助损失函数实际上可以防止过拟合。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/09_fine_tune_evals.png" alt="与稠密模型相比，稀疏模型从指令微调中受益更多">
  <figcaption>与稠密模型相比，稀疏模型从指令微调中受益更多。该图来自 MoEs Meets instructions Tuning 论文。</figcaption>
</figure>

## 何时使用稀疏 MoE，何时使用稠密模型？

MoE 非常适合多机高吞吐量的场景。在给定预训练算力预算的情况下，训得的稀疏模型效果更优。对于内存较少的低吞吐场景，稠密模型会更好。

**注意：** 不能直接对稀疏模型和稠密模型的参数量进行比较，因为两者的意义迥异。

## 让 MoE 性能飞起

最初大家用分支逻辑来实现 MoE 层，这会导致计算速度变慢，因为 GPU 并非为此设计，并且当当前设备需要向其他设备发送信息时，网络带宽会成为瓶颈。本节将讨论现有的能加速这些模型的预训练和推理的方法。让 MoE 性能飞起！

### 并行

我们简单回顾一下现有的并行方案：

- **数据并行：**所有工作进程（或线程）都各自保留一份完整的模型权重，每个工作进程（或线程）负责处理同一 batch 的不同数据。
- **模型并行：**跨工作进程（或线程）划分模型权重，每个工作进程（或线程）使用同一批数据。
- **模型和数据并行：**跨工作进程（或线程）对数据和模型均进行划分。请注意，此时每个工作进程（或线程）会被分配到同一 batch 的不同数据。
- **专家并行**：不同的专家被分配至不同的工作进程（或线程）。当其与数据并行同时使用时，每个工作进程（或线程）上的专家不同，且每个工作进程（或线程）处理同一 batch 的不同数据。

在专家并行方案中，不同的专家被安排到不同的工作进程（或线程）上，每个工作进程（或线程）负责处理同一 batch 的不同数据。对于非 MoE 层，其行为与数据并行相同，当数据到了 MoE 层时，序列中的词元会被发送给相应专家所在的工作进程（或线程）。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/10_parallelism.png" alt="模型并行、专家并行及数据并行示意图">
  <figcaption>上图来自 Switch Transformers 论文，其展示了如何使用不同的并行技术在各工作进程（或线程）间划分数据和模型。</figcaption>
</figure>

### 容量因子和通信开销

增加容量因子 (capacity factor，CF) 可提高模型质量，但会增加通信成本以及内存（用于存储激活）。如果 all-to-all 通信速度较慢，则使用较小的容量因子会更好。一个不错的初始配置是，容量因子为 1.25 的 top-2 路由，且每个工作进程分配（线程）1 个专家。在推理时，可以更改容量因子以减少计算量。

### 推理部署技术

> 你可以在推理终端上部署 [mistralai/Mixtral-8x7B-Instruct-v0.1](https://ui.endpoints.huggingface.co/new?repository=mistralai%2FMixtral-8x7B-Instruct-v0.1&vendor=aws&region=us-east-1&accelerator=gpu&instance_size=2xlarge&task=text-generation&no_suggested_compute=true&tgi=true&tgi_max_batch_total_tokens=1024000&tgi_max_total_tokens=32000) 模型。 

MoE 的一大缺点是参数量过大。对于本地推理而言，大家可能想要模型变小点。我们快速总结了当前一些有助于推理部署的工作：

* Switch Transformers 的作者做了一些初步的蒸馏实验。通过将 MoE 蒸馏成一个稠密模型，能够保留 30-40% 的稀疏性增益。因此，预训练一个稀疏模型再蒸馏成稠密模型，可以两全其美，得到更快的预训练速度以及更快的生产推理。
* 最近还出现了修改路由以支持将完整的句子或任务路由给专家的做法，该做法使得提取子网络进行推理成为可能。
* 专家聚合：该技术对专家的权重进行了合并，从而减少了推理时的参数量。

### 高效训练

FasterMoE（2022 年 3 月）这一工作分析了高效分布式系统中 MoE 的性能，并分析了不同并行策略的理论极限，以及各种加速技术，如处理专家欢迎度偏倚的技术、用于减少延迟的细粒度通信调度技术以及根据网络延迟来选择专家的拓扑感知的门控机制等，这些技术总共带来了 17 倍的训练加速。

Megablocks（2022 年 11 月）则开发了一系列新的能高效处理 MoE 动态性的GPU 核函数从而实现高效的稀疏预训练。这种做法不会丢失词元并能有效利用当前硬件的能力，从而显著加快速度。窍门是什么呢？传统的 MoE 假设所有专家处理相同的形状以及相同数量的词元，并使用批量矩阵乘来进行计算。相比之下，Megablocks 将 MoE 层实现为块稀疏操作，从而对专家分配不均衡场景更高效。

<figure class="image text-center">
  <img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/moe/11_expert_matmuls.png" alt="针对块稀疏操作优化过的矩阵乘">
  <figcaption>针对不同大小的专家和词元数量的块稀疏矩阵乘 (来自论文 [MegaBlocks](https://arxiv.org/abs/2211.15841))。</figcaption>
</figure>

## 开源 MoE 项目

以下列出了当前几个训练 MoE 模型的开源项目：

- Megablocks: https://github.com/stanford-futuredata/megablocks
- Fairseq: https://github.com/facebookresearch/fairseq/tree/main/examples/moe_lm
- OpenMoE: https://github.com/XueFuzhao/OpenMoE

目前有下面这些开放 MoE 模型可供使用：

- [Switch Transformers (Google)](https://huggingface.co/collections/google/switch-transformers-release-6548c35c6507968374b56d1f)：基于 T5 的 MoE 模型族，专家数从 8 到 2048 不等，其中最大的模型有 1.6 万亿参数。
- [NLLB MoE (Meta)](https://huggingface.co/facebook/nllb-moe-54b): NLLB 翻译模型的 MoE 版。
- [OpenMoE](https://huggingface.co/fuzhao): 社区发布的 Llama 模型的 MoE 版。
- [Mixtral 8x7B (Mistral)](https://huggingface.co/mistralai): 高质量 MoE 模型，其性能优于 Llama 2 70B，并且推理速度更快。Mistral 还发布了其指令微调版模型。欲了解详情，可阅读[这篇博文](https://mistral.ai/news/mixtral-of-experts/)。

## 后续方向

进一步探索将稀疏 MoE **蒸馏**为等效的稠密模型的方法。

另一个方向是 MoE 的量化。[QMoE](https://arxiv.org/abs/2310.16795)（2023 年 10 月）通过将 MoE 量化至每参数不足 1 比特，从而将 1.6T 的 Switch Transformer 模型的内存使用量从 3.2TB 压缩至仅需 160GB。

总结一下，下面是几个我们觉得值得探索的有意思的领域：

* 将 Mixtral 蒸馏成稠密模型
* 探索专家合并技术及其对推理时间的影响
* 对 Mixtral 执行极致量化

## 资源

- [Adaptive Mixture of Local Experts (1991)](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)
- [Learning Factored Representations in a Deep Mixture of Experts (2013)](https://arxiv.org/abs/1312.4314)
- [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer (2017)](https://arxiv.org/abs/1701.06538)
- [GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding (Jun 2020)](https://arxiv.org/abs/2006.16668)
- [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts (Dec 2021)](https://arxiv.org/abs/2112.06905)
- [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity (Jan 2022)](https://arxiv.org/abs/2101.03961)
- [ST-MoE: Designing Stable and Transferable Sparse Expert Models (Feb 2022)](https://arxiv.org/abs/2202.08906)
- [FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models(April 2022)](https://dl.acm.org/doi/10.1145/3503221.3508418)
- [MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (Nov 2022)](https://arxiv.org/abs/2211.15841)
- [Mixture-of-Experts Meets Instruction Tuning:A Winning Combination for Large Language Models (May 2023)](https://arxiv.org/abs/2305.14705)
- [Mixtral-8x7B-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1), [Mixtral-8x7B-Instruct-v0.1](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1).


## 本文引用格式

```bibtex
@misc {sanseviero2023moe,
    author       = { Omar Sanseviero and
                     Lewis Tunstall and
                     Philipp Schmid and
                     Sourab Mangrulkar and
                     Younes Belkada and
                     Pedro Cuenca
                   },
    title        = { Mixture of Experts Explained },
    year         = 2023,
    url          = { https://huggingface.co/blog/moe },
    publisher    = { Hugging Face Blog }
}
```

```
Sanseviero, et al., "Mixture of Experts Explained", Hugging Face Blog, 2023.
```

> 英文原文: <url> https://huggingface.co/blog/moe </url>
> 原文作者：Omar Sanseviero，Lewis Tunstall，Philipp Schmid，Sourab Mangrulkar，Younes Belkada，Pedro Cuenca
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。
