> Lastest Update: 9th May 2024&nbsp

[Chatgpt_documentaion](https://platform.openai.com/docs/introduction)
# Get started
## 介绍-introduction
### 关键概念-Key concepts
#### Text generation models
```
OpenAI的文本生成模型(通常被称为生成式预训练transformer或简称“GPT”模型)，如GPT-4和GPT-3.5，已被训练以理解自然和formal语言。像GPT-4这样的模型允许根据输入,输出文本。这些模型的输入也称为“prompt”。设计prompt符本质上是如何“编程”像GPT-4这样的模型，通常通过提供如何成功完成任务的说明或一些示例。像GPT-4这样的模型可以用于各种任务，包括内容或代码生成、摘要、对话、创意写作等。更多信息请阅读我们的入门文本生成指南和提示工程指南。
```
#### Assistants
```
助手指的是实体(entity)，在OpenAI API中，它们由GPT-4等大型语言模型提供支持，能够为用户执行任务。这些助手基于嵌入在模型上下文窗口中的指令进行操作。他们通常还可以访问工具，这些工具允许助手执行更复杂的任务，如运行代码或从文件中检索信息。在我们的助手API概述中阅读有关助手的更多信息。
```
#### Embeddings
```
嵌入是一段数据(例如一些文本)的向量表示，旨在保留其内容和/或含义的某些方面。在某种程度上相似的数据块往往比不相关的数据具有更紧密的嵌入关系。OpenAI提供文本嵌入模型，将文本字符串作为输入，并产生一个嵌入向量作为输出。嵌入在搜索、聚类、推荐、异常检测、分类等方面很有用。在我们的嵌入指南中阅读有关嵌入的更多信息。
```
#### Tokens
```
文本生成和嵌入模型以称为“token”的块,处理文本。token表示常见的字符序列。例如，字符串“tokenization”被分解为“token”和“ization”，而像“the”这样的短而常见的单词被表示为单个标记。注意，在句子中，每个单词的第一个单词通常以空格字符开头。请查看我们的tokenizer工具，以测试特定的字符串，并查看它们如何转换为标记。作为一个粗略的经验法则，在英语文本中，1个标记大约等于4个字符或0.75个单词。

要记住的一个限制是，对于文本生成模型，prompt和生成的输出组合不得超过模型的最大上下文长度。对于嵌入模型(不输出tokens)，输入必须小于模型的最大上下文长度。每个文本生成和嵌入模型的最大上下文长度可以在模型索引中找到。
```

## 快速开始-Quickstart
To be done


## 模型-Models
### 概述-overview
OpenAI API由一组具有不同功能和价格点的不同模型驱动。您还可以通过微调对我们的模型进行自定义，以满足您的特定用例。
|MODEL|	DESCRIPTION|翻译|
|---|---|---|
|GPT-4 Turbo and GPT-4	|A set of models that improve on GPT-3.5 and can understand as well as generate natural language or code|一组改进了GPT-3.5的模型，可以理解和生成自然语言或代码|
|GPT-3.5 Turbo	|A set of models that improve on GPT-3.5 and can understand as well as generate natural language or code|一组改进了GPT-3.5的模型，可以理解和生成自然语言或代码|
|DALL·E	|A model that can generate and edit images given a natural language prompt|在给定自然语言提示的情况下，可以生成和编辑图像的模型|
|TTS	|A set of models that can convert text into natural sounding spoken audio|一套可以将文本转换为自然发音的语音的模型|
|Whisper	|A model that can convert audio into text|一个可以将音频转换为文本的模型|
|Embeddings	|A set of models that can convert text into a numerical form|一组可以将文本转换为数字形式的模型|
|Moderation	|A fine-tuned model that can detect whether text may be sensitive or unsafe|一种微调模型，可以检测文本是否敏感或不安全|
|GPT base	|A set of models without instruction following that can understand as well as generate natural language or code|一组无需遵循指令就能理解和生成自然语言或代码的模型|
|Deprecated	|A full list of models that have been deprecated along with the suggested replacement|已弃用的模型的完整列表，以及建议的替换方案|

除此之外OPENAI还发布了开源模型，包括Point-E、Whisper、Jukebox和CLIP。

### Continuous model upgrades
gpt-4-turbo、gpt-4和gpt-3.5-turbo分别对应各自的最新模型版本。您可以在发送请求后查看response对象来验证这一点。响应将包括使用的特定型号版本(例如gpt-3.5-turbo-0613)。

我们还提供固定模型版本，在引入更新的模型后，开发人员可以继续使用至少三个月。随着模型更新的新节奏，我们也让人们能够贡献评估，以帮助我们改进针对不同用例的模型。如果您感兴趣，请查看OpenAI eval存储库。

有关模型弃用的更多信息，请访问我们的弃用页面。

### GPT-4 Turbo and GPT-4

GPT-4是一个大型多模态模型(接受文本或图像输入并输出文本)，由于其更广泛的一般知识和高级推理能力，它可以比我们之前的任何模型更准确地解决困难问题。GPT-4可在OpenAI API中向付费客户提供。像gpt-3.5-turbo一样，GPT-4针对聊天进行了优化，但对于使用聊天完成API的传统完成任务效果很好。在我们的文本生成指南中学习如何使用GPT-4。

GPT-4 - Turbo
(新)带视觉的GPT-4 Turbo
具有视觉能力的最新GPT-4 Turbo模型。视觉请求现在可以使用JSON模式和函数调用。目前指向gpt-4-turbo-2024-04-09。
数据截至2023年12月，
窗口大小共有128,000token
<!-- |MODEL	|DESCRIPTION	|CONTEXT WINDOW	|TRAINING DATA|
|---|---|---|---|
|gpt-4-turbo|	(New) GPT-4 Turbo with Vision <br>The latest GPT-4 Turbo model with vision capabilities. Vision requests can now use JSON mode and function calling. Currently points to gpt-4-turbo-2024-04-09.	|128,000 tokens	|Up to Dec 2023|

gpt-4-turbo-2024-04-09	GPT-4 Turbo with Vision model. Vision requests can now use JSON mode and function calling. gpt-4-turbo currently points to this version.	128,000 tokens	Up to Dec 2023
gpt-4-turbo-preview	GPT-4 Turbo preview model. Currently points to gpt-4-0125-preview.	128,000 tokens	Up to Dec 2023
gpt-4-0125-preview	GPT-4 Turbo preview model intended to reduce cases of “laziness” where the model doesn’t complete a task. Returns a maximum of 4,096 output tokens. Learn more.	128,000 tokens	Up to Dec 2023
gpt-4-1106-preview	GPT-4 Turbo preview model featuring improved instruction following, JSON mode, reproducible outputs, parallel function calling, and more. Returns a maximum of 4,096 output tokens. This is a preview model. Learn more.	128,000 tokens	Up to Apr 2023
gpt-4-vision-preview	GPT-4 model with the ability to understand images, in addition to all other GPT-4 Turbo capabilities. This is a preview model, we recommend developers to now use gpt-4-turbo which includes vision capabilities. Currently points to gpt-4-1106-vision-preview.	128,000 tokens	Up to Apr 2023
gpt-4-1106-vision-preview	GPT-4 model with the ability to understand images, in addition to all other GPT-4 Turbo capabilities. This is a preview model, we recommend developers to now use gpt-4-turbo which includes vision capabilities. Returns a maximum of 4,096 output tokens. Learn more.	128,000 tokens	Up to Apr 2023
gpt-4	Currently points to gpt-4-0613. See continuous model upgrades.	8,192 tokens	Up to Sep 2021
gpt-4-0613	Snapshot of gpt-4 from June 13th 2023 with improved function calling support.	8,192 tokens	Up to Sep 2021
gpt-4-32k	Currently points to gpt-4-32k-0613. See continuous model upgrades. This model was never rolled out widely in favor of GPT-4 Turbo.	32,768 tokens	Up to Sep 2021
gpt-4-32k-0613	Snapshot of gpt-4-32k from June 13th 2023 with improved function calling support. This model was never rolled out widely in favor of GPT-4 Turbo.	32,768 tokens	Up to Sep 2021

For many basic tasks, the difference between GPT-4 and GPT-3.5 models is not significant. However, in more complex reasoning situations, GPT-4 is much more capable than any of our previous models. -->

对于许多基本任务，GPT-4和GPT-3.5模型之间的差异并不显著。然而，在更复杂的推理情况下，GPT-4比之前的任何模型都更有能力。