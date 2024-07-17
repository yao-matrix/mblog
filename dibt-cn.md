---
title: "数据好合：回望与展望" 
thumbnail: /blog/assets/dibt/thumbnail.png
authors:
- user: davanstrien
- user: davidberenstein1957
- user: sdiazlor
- translators:
- user: MatrixYao
---

# 数据好合：回望与展望

过去的几个月里，我们一直致力于[数据好合](https://github.com/huggingface/data-is-better-together)计划。我们的目标是：通过 Hugging Face 和 Argilla 的合作以及开源机器学习社区的支持，赋能开源社区共建有影响力的数据集。

现在，我们决定朝着既定目标继续前进。在此，我们简要向大家报告一下我们已有的成就和后续的任务，以确保人人都对其有一定的了解并能够为此作出贡献。为清晰起见，下文将我们的工作分为两个部分：社区工作和攻略工作。

## 社区工作

此项工作的首要重点是**提示词排名**项目。其目标是创建一个包含 10K 个提示的数据集，包括合成提示以及人工生成提示，并对它们按质量进行排名。社区很快对此作出了响应！

- 仅用几天，就有超 385 人加入。
- 我们发布了 [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) 数据集，以用于提示排名任务或生成合成数据。
- 该数据集已被用于训练新的[模型](https://huggingface.co/models?dataset=dataset:DIBT/10k_prompts_ranked)，如 SPIN。

我们发现社区支持来自全球，因此意识到仅以英语为中心的数据是不够的，我们现在缺乏足够的针对特定语言的开放 LLM 基准。因此，我们创建了**多语种提示词评估项目（Multilingual Prompt Evaluation Project，MPEP）**，旨在开发多语种的排行榜。为此，我们从 [DIBT/10k_prompts_ranked](https://huggingface.co/datasets/DIBT/10k_prompts_ranked) 中精选了 500 个高质量提示词，将它们翻译成不同的语言。

- 超过 18 位语言领袖为此翻译工作创建了 space。
- 已完成[荷兰语](https://huggingface.co/datasets/DIBT/MPEP_DUTCH)、[俄语](https://huggingface.co/datasets/DIBT/MPEP_RUSSIAN)以及[西班牙语](https://huggingface.co/datasets/DIBT/MPEP_SPANISH)的翻译工作。为全面完成翻译工作，更多工作正在紧锣密鼓地进行中。
- 在 Discord 上建立了数据集构建者社区。

展望未来，我们将致力于持续完善工具和文档，以支持社区构建更多开放数据集。 

## 攻略工作

作为 [DIBT](https://github.com/huggingface/data-is-better-together) 计划的一部分，我们还创建了指南及工具以帮助社区自行构建有价值的数据集。

- **特定领域数据集**：将工程师和领域专家聚拢到一起，为创建更多特定领域数据集做一些开路性质的工作，以助力领域模型的训练。
- **DPO/ORPO 数据集**：助力培育社区，为不同语种、领域和任务构建更多 DPO 风格的数据集。
- **KTO 数据集**：助力社区创建自己的 KTO 数据集。

## 我们学到了什么？

- 社区渴望参与这些工作，并且对集智工作在同一个数据集上感到兴奋。
- 必须克服现有的不平等现象，以确保基准的全面性和包容性。目前，某些语种、领域及任务的数据集在开源社区中的代表性不足。
- 我们拥有社区所需的很多工具，可以借助它们有效地协作构建有价值的数据集。

## 你如何参与？

如你想参与攻略相关的工作，你可以按照你感兴趣的项目的 README 文件中的说明进行操作，与社区共享你的数据集和结果，或者为大家提供新的指南和工具。你的宝贵贡献将有助于我们为所有人建立强大而全面的资源。

如果你想参与其中，请加入我们的 [**Hugging Face Discord**](http://hf.co/join/discord) 中的`#data-is-better-together` 频道，并告诉我们你想参与什么任务！

我们期待与大家一起构建更好的数据集！

> 英文原文: <url> https://huggingface.co/blog/dibt </url>
> 原文作者：Daniel van Strien，David Berenstein，Sara Han Díaz
> 译者: Matrix Yao (姚伟峰)，英特尔深度学习工程师，工作方向为 transformer-family 模型在各模态数据上的应用及大规模模型的训练推理。