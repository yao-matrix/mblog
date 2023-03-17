# 如何评估大语言模型

大家可以使用 Hugging Face Space 上的[Evaluation on the Hub](https://huggingface.co/spaces/autoevaluate/model-evaluator) 应用在零样本分类任务上评估大语言模型啦！

零样本评估是研究人员衡量大语言模型性能的一种流行方法。因为大语言模型已经在训练过程中[显示](https://arxiv.org/abs/2005.14165) 出了对无标注数据的学习能力。[反向缩放奖](https://github.com/inverse-scaling/prize) 是近期社区的一项工作，通过在各种尺寸和结构的模型上进行大规模零样本评估，以发现哪些大模型的表现比小模型还差。

![数据集](assets/106_zero_shot_eval_on_the_hub/zeroshot.jpg)

## 在 Hub 上零样本评估语言模型

[Evaluation on the Hub](https://huggingface.co/blog/eval-on-the-hub) 无需编写代码即可帮助你评估 Hub 上的任何模型，这个能力是由 [AutoTrain](https://huggingface.co/autotrain) 赋予的。现在，Hub 上的任何因果语言模型都可以以零样本的方式进行评估。零样本评估会度量训得的模型生成一组特定补全词的可能性，且不需要任何标注训练数据，这使得研究人员可以跳过昂贵的标注工作。

我们已经为该项目升级了 AutoTrain 基础设施，使之可以免费评估大模型🤯！用户弄清楚如何自己编写可用于GPU执行的评估再运行该代码去评估大模型既昂贵又耗时。例如，一个具有 660 亿参数的语言模型可能仅加载和编译就需要 35 分钟，这使得大模型的评估只有那些拥有昂贵的基础设施和丰富的技术经验的人才能进行。通过升级 AutoTrain 基础设施，在具有 2000 个句子的零样本分类任务中评估一个 660 亿参数的模型需要 3.5 小时，社区中的任何人都可以完成。 Evaluation on the Hub 目前支持评估的最大模型尺寸为 660 亿参数，后面会支持更大的模型。

零样本文本分类任务采用包含一组提示和及其候选补全的数据集。在工作时，会将补全与提示连接起来，并对每个补全的对数概率求和，然后进行归一化并与正确的补全进行比较以最终输出任务的准确性。

在本文中，我们将在 [WinoBias](https://uclanlp.github.io/corefBias/overview) 数据集上来评估各种 [OPT](https://ai.facebook.com/blog/democratizing-access-to-large-scale-language-models-with-opt-175b/) 模型在零样本文本分类任务上的表现，该任务是一个度量职业上的性别偏见的共指任务。 WinoBias 度量一个模型是否更有可能选择一个带有刻板印象的代词来填充一个提到职业的句子。我们通过观察结果发现，在此任务上模型效果与模型大小存在[反向缩放](https://github.com/inverse-scaling/prize)，即模型越大越容易产生带有刻板印象的填充代词。

## 案例研究：WinoBias 任务的零样本评估

[WinoBias](https://github.com/uclanlp/corefBias) 数据集已被我们转换成适合零样本任务所需的格式，其任务就是视不同的补全为不同的类（下图中的 `classes` 列），并进行预测。每个补全句的区别在于代词不同，而标签（下图中的 `target` 列）就是对应职业的反刻板化补全（例如，“开发人员”通常是男性主导的职业，因此“她”就是反刻板化代词）。有关示例，请参见[此处](https://huggingface.co/datasets/mathemakitten/winobias_antistereotype_test)。

![数据集](assets/106_zero_shot_eval_on_the_hub/dataset.png)

接下来，我们可以在 `Evaluation on the Hub` 界面上将任务设置为`text_zero_shot_classification`，并选择 winobias 数据集（见下图），然后选择我们想要评估的模型，最后提交评估任务！评估完成后，你将收到电子邮件通知，`autoevaluator` 机器人会在模型的 Hub 库中新建一个 PR，并把结果放在这个 PR 里。

![Evaluation on the Hub 界面](assets/106_zero_shot_eval_on_the_hub/eval_hub.png)

下图是 WinoBias 任务的结果，我们发现较小的模型更有可能生成反刻板化的补全，而较大的模型更有可能学到文本中性别和职业之间的刻板化关联。这与其他基准（例如 [BIG-Bench](https://arxiv.org/abs/2206.04615)）的结果一致，这些基准也表明更大、更强的模型更有可能在性别、种族、民族、和国籍问题上形成偏见。另外，[之前的工作](https://www.deepmind.com/publications/scaling-language-models-methods-analysis-insights-from-training-gopher) 也表明较大的模型更有可能产生有毒文本。

![Winobias](./assets/106_zero_shot_eval_on_the_hub/winobias.png)

## 为每个人提供更好的研究工具

开放科学在社区驱动的工具开发方面取得了长足进步，例如 EleutherAI 的 [语言模型评估工具](https://github.com/EleutherAI/lm-evaluation-harness) 和 [BIG-bench](https://github.com/google/BIG-bench) 项目，这使得研究人员可以直观了解最先进模型的行为。

`Evaluation on the Hub` 是一种低代码工具，让研究人员可以很容易地按照某个维度（例如 FLOPS 或模型大小）比较一组模型的零样本性能，或者比较在同一语料库上训出来的一组不同的模型的性能。零样本文本分类任务非常灵活 —— 任何可以转换成 Winograd 格式的数据集（该格式中要比较的句子间仅有几个单词不同），都可以用于此任务，并可以同时评估多个模型。我们的目标是让研究人员能够很容易地上传新数据集，并能轻松地在其上对许多模型进行基准测试。

一个可以用该工具解决的研究问题的例子是反向缩放问题：虽然较大的模型通常在大多数语言任务上表现更强，但在某些任务中较大的模型表现反而较差。[反向缩放奖](https://github.com/inverse-scaling/prize) 作为一个比赛，旨在鼓励研究人员去构建一些任务，在这些任务上大模型比小模型表现更差。我们鼓励你在自己的任务上尝试对各种尺寸的模型进行零样本评估！如果你发现你的任务效果与模型尺寸的相对关系比较有趣，我们可以聊一聊。

## 向我们提交反馈！

在 Hugging Face，我们很高兴能通过我们的持续工作，让人人都能访问到最先进的机器学习模型，包括开发工具，并让每个人都能轻松评估和探究这些模型的行为。我们之前已经[强调过](https://huggingface.co/blog/eval-on-the-hub)可保证评估结果一致性和可重复性的标准化模型评估方法以及使评估工具人人可访问的重要性。Evaluation on the Hub 的未来计划包括支持对其他语言任务的零样本评估，这些任务可能不适合使用“提示+补全”的数据格式，同时我们还会增加对更大模型的支持。

作为社区的一部分，向我们提交反馈十分有助于我们的工作！我们很乐意听取你关于模型评估的首要任务是什么的意见。通过 Hub [社区](https://huggingface.co/spaces/autoevaluate/model-evaluator/discussions)或[论坛](https://discuss.huggingface.co/)告诉我们吧!

> 英文原文: <url> https://huggingface.co/blog/zero-shot-eval-on-the-hub </url>
> 原文作者：Mathemakitten Helen，Tristan Thrush，Abhishek Thakur，Lewis Tunstall，Douwe Kiela
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。