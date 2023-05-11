---
title: "〜不要〜重复自己"
thumbnail: /blog/assets/59_transformers_philosophy/transformers.png
authors:
- user: patrickvonplaten
translators:
- user: MatrixYao
---

<h1>
	<del>不要</del>重复自己*
	<h5><i> 如何为现代机器学习设计开源库 </i></h5>
</h1>

<!-- {blog_metadata} -->
<!-- {authors} -->

## 🤗 Transformers 设计理念

*“不要重复自己（Don't Repeat Yourself）”*，或 **DRY**，是众所周知的软件开发原则。该原则源自《程序员修炼之道：从小工到专家》（英文名为 The pragmatic programmer），这是一本代码设计领域迄今阅读量最大的书籍。该原则言简意赅，即：重用而不要重写其他地方已有的逻辑。这可以确保代码保持同步，使其更易于维护且更健壮。这样的话，对该公共代码逻辑的任何更改都会统一地影响所有依赖它的代码。

乍一看，Hugging Face Transformers 库的设计与 DRY 原则背道而驰。注意力机制的代码被复制到不同的模型文件里不下 50 多次。有时整个 BERT 模型的代码也会被复制到其他模型文件中。我们经常强制贡献者在增加新模型时，如果该模型使用了已存在的模型，需要复制现有模型的所有代码，连一个小小的逻辑调整也不例外。我们为什么要这么做？是因为我们太懒抑或是因为我们无力承担将所有公共逻辑集中到一个地方的工作？

不，我们并不懒 —— 不将 DRY 设计原则应用于 transformers 库是有意之举。相反，我们决定采用一种不同的设计原则，我们称之为 ***单一模型文件*** 策略。 *单一模型文件*策略要求，模型前向传播所需的所有代码都在一个且仅在一个文件中 —— 即模型文件。如果读者想了解 BERT 如何进行推理，他/她只需要看 BERT 的 `modeling_bert.py` 文件即可。我们通常拒绝任何将不同模型的相同子模块集中抽象到一个新地方的尝试。我们不想要一个包含所有可能的注意力机制的 `attention_layer.py`。那我们为什么要这样做呢？

简而言之，原因如下：
- **1. Transformers 由开源社区构建并为开源社区服务。**
- **2. 我们的产品是模型，我们的客户是阅读或调整模型代码的用户。**
- **3. 机器学习领域发展极其迅速。**
- **4. 机器学习模型是静态的。**

### 1. 由开源社区构建并为开源社区服务

Transformers 旨在积极激励外部贡献。贡献通常有错误修复和新模型添加。如果在其中一个模型文件中发现错误，我们希望发现者尽可能容易地修复它。没有什么比修复一个 bug 却看到它导致了100 个其他模型上的失败更令人沮丧的了。

因为模型代码独立于所有其他模型，所以对于只了解他/她正在使用的一个模型的人来说，修复它会比较容易。同样，如果只添加一个新的模型文件，添加新的模型代码以及 review 相应的 PR 会更容易。贡献者不必弄清楚如何在不破坏现有模型的情况下向公共的注意力机制代码添加新功能。Reviewer 也缺省地知道没有任何一个现有模型被破坏。

### 2. 模型代码即产品

我们假设 Transformers 库的大量用户不仅会阅读文档，而且会查看实际模型代码并可能对其进行修改。这个假设是成立的，因为我们的 Transformers 库被 fork 了 1 万多次，我们的 Transformers 论文被引用了 1 千多次。

因此，最重要的是让第一次阅读 Transformers 建模代码的人能够轻松理解并适配它。在单个建模文件中按顺序提供所有必要的逻辑组件有助于提高可读性和可适配性。此外，我们非常关心变量及方法命名的合理性，并且更倾向于表达力强/可读性高的代码，而不追求代码长度短。

### 3. 机器学习正以惊人的速度发展
机器学习领域，尤其是神经网络领域的研究发展非常迅速。一年前最先进的模型今天可能已经过时了。我们甚至不知道明年哪一种注意力机制、位置嵌入或架构会是最佳。因此，我们无法定义适用于所有模型的标准逻辑模板。

例如，两年前，人们可能将 BERT 的自注意力层定义为所有 transformer 模型使用的标准注意力层。从逻辑上讲，“标准”注意力函数可以移到一个集中性的 `attention.py` 文件中。但是随后出现了在每层中添加相对位置嵌入的注意力层（如 T5），多种不同形式的分块注意力层（Reformer，Longformer，BigBird），以及将位置嵌入和词嵌入分离的注意力机制（DeBERTa）等......每当这样的事情发生时，我们都不得不问自己是否应该调整“标准”注意力函数，还是说向 `attention.py` 添加一个新的注意力函数更好。但如果要添加新的注意力函数，我们该如何命名呢？ `attention_with_positional_embd`，`reformer_attention` 以及 `deberta_attention`？

给机器学习模型的组件起通用的名字是危险的，因为对该组件的代表什么的看法可能会很快改变或过时。例如，分块注意力指的是 GPTNeo 的分块注意力，还是 Reformer 的分块注意力，抑或是 BigBird 的分块注意力？注意层是自注意层、交叉注意层，还是两者都包含？但是，如果我们用模型名称命名注意力层，我们应该直接将注意力函数放在相应的模型文件中。

### 4. 机器学习模型是静态的
Transformers 库是不同研究团队创建的统一且完善的机器学习模型的集合。每个机器学习模型通常都对应一篇论文及其官方 GitHub 存储库。机器学习模型一旦发布，之后很少会进行调整或更改。

相反，研究团队倾向于发布基于之前模型构建的新模型，但很少对已发布的代码进行重大更改。在决定 Transformers 库的设计原则时，这是一个重要的认知。这意味着一旦将模型架构添加到 Transformers 中，模型的基本组件就不会再改变。通常会发现并修复错误，可能会重命名方法和变量，并且模型的输出或输入格式可能会略有更改，但模型的核心组件不会再更改。因此，对 Transformers 中的所有模型进行大的全局更改的需求大大减少，这使得每个逻辑模块只存在一次变得不那么重要，因为我们很少会更改它。

第二个认知是模型之间**不**双向依赖。最近发布的模型可能依赖于现有模型，但很明显，现有模型在逻辑上不能依赖于其后继模型。例如，T5 部分建立在 BERT 之上，因此 T5 的模型代码在逻辑上可能依赖于 BERT 的模型代码，但 BERT 在逻辑上不能以任何方式依赖于 T5。因此，重构 BERT 的注意力功能以使其可以适用于 T5 在逻辑上是不合理的 —— 阅读 BERT 的注意力层代码的人不需要对 T5 有任何了解。同样，这也促使我们不要将注意力层等组件集中到所有模型都可以访问的模块中。

另一方面，后继模型的模型代码在逻辑上可以很好地依赖于其前代模型。例如，DeBERTa-v2 建模代码在逻辑上确实在某种程度上依赖于 DeBERTa 的模型代码。通过确保 DeBERTa-v2 的模型代码与 DeBERTa 的保持同步，可维护性可以得到显著提高。理论上来讲，修复 DeBERTa 中的 bug 应该同时也修复 DeBERTa-v2 中的相同 bug。我们如何在确保后继模型与其前代模型保持同步的同时维持*单一模型文件*策略？

现在，我们解释一下为什么我们在 *“重复自己”* 之后加上星号$ {}^{\textbf{*}} $。我们不会盲目地复制粘贴所有现有的模型代码，即使看上去我们好像是这么做的。 Transformers 的核心维护者之一 [Sylvain Gugger](https://github.com/sgugger) 发现了一种既尊重 *单文件策略* 又将可维护性成本控制在一定范围内的好机制。这种机制，我们暂且称其为 *“复制机制”*，允许我们使用 `#Copied from <predecessor_model>.<function>` 语句标记某些逻辑组件，如注意力层函数，从而强制标记代码与 `<predecessor_model>` 的 `<function>` 相同。例如，[DeBERTa-v2 类](https://github.com/huggingface/transformers/blob/21decb7731e998d3d208ec33e5b249b0a84c0a02/src/transformers/models/deberta_v2/modeling_deberta_v2.py#L325) 里的这行代码强制除了类名前缀`DeBERTav2` 之外的整个类与 [DeBERTa 类](https://github.com/huggingface/transformers/blob/21decb7731e998d3d208ec33e5b249b0a84c0a02/src/transformers/models/deberta/modeling_deberta.py#L336)相同。如此可以看到，复制机制使模型代码非常容易理解，同时又显著了减少维护成本。如果在其后继模型的函数引用的前代模型的函数中更改了某些代码，则可以使用适当的工具自动更正后继模型的中的相应代码。

### 缺点

显然，单文件策略也有缺点，我们在这里快速提两个。

Transformers 的一个主要目标是为所有模型的推理和训练提供统一的 API，以便用户可以在不同模型之间快速切换。但是，如果不允许模型文件使用抽象的逻辑模式，则确保跨模型的统一 API 会困难得多。我们通过运行**大量**测试（*ca.* 截至本文撰写时每天需要运行大约 20,000 次测试）来解决这个问题，以确保模型遵循一致的 API。在这种情况下，单文件策略要求我们在 review 新模型和新测例时非常严格。

其次，有很多研究仅针对机器学习模型的单个组件。 *例如*，研究团队正在研究适用于所有现有预训练模型的注意力机制的新形式，如 [Rethinking Attention with Performers](https://arxiv.org/abs/2009.14794) 中所做。我们应该如何将此类研究纳入 transformers 库？确实不好弄。我们应该改变所有现有模型吗？这将违背上文中的第 3 点和第 4 点。还是我们应该添加 100 多个新的模型文件，每个文件都以 `Performer...` 为前缀？这也很荒谬。遗憾的是，对此类情况我们还没有好的解决方案，我们只能选择不将该论文的成果集成到 Transformers 中。如果这篇论文获得更多关注并有了性能强大的预训练 checkpoint，我们可能会为其中最重要的模型增加一个新的模型文件，例如目前已有 `modeling_performer_bert.py`。

### 总结
总而言之，在 🤗 Hugging Face，我们坚信*单一文件策略*是适合 Transformers 的代码设计理念。

你的想法如何？我们很想听听你的意见！如果你想发表评论，欢迎到相应的论坛[帖子](https://discuss.huggingface.co/t/repeat-yourself-transformers-design-philosophy/16483)下留言。

> 英文原文: <url> https://huggingface.co/blog/transformers-design-philosophy </url>
> 原文作者：Patrick von Platen
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。