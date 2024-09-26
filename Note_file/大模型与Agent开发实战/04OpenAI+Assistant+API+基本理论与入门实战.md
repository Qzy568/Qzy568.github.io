# <center>大模型 AI Agent 开发实战

## <center>Ch.4 OpenAI Assistant API 基本理论与入门实战

&emsp;&emsp;在深入探讨了 `OpenAI` 的 `Function Calling` 和 `ReAct` 的基础理论及相关实战案例后，在接下来的课程中，我们将逐步进入一个个全新的 `AI Agent` 应用开发框架的学习中。

&emsp;&emsp;就目前我们对`AI Agent`工程化开发的不断探索中，会认为在这个技术领域普遍存在的问题是：**理论多但落地难的难题**。有非常多的技术思想试图引导大模型进行自主推理从而实现高级的智能体应用，但**很多都停留在理论层，在真实场景中使用起来效果并不是很明显，且过程不可控。各个`AI Agent`开发框架看似百花齐放，实则都是基于一套类似的理论在各自擅长的领域进行扩展。**`Function Calling` 和 `ReAct` 作为目前热门`AI Agent`框架/理论的基础，具有非常强的启发性，掌握它们对于后续其他`AI Agent`开发框架的学习都是非常有必要的，能帮助大家更快、更好的理解各个框架的底层原理。同时这也是我们在`AI Agent`课程的第一阶段重点给大家介绍这两个技术点的根本原因。

&emsp;&emsp;除此之外，需要向大家说明的是**，`OpenAI`对大模型整体技术生态发展的影响力远远不限于`GPT`模型本身**，就像我们一直开玩笑提到的，大模型只分为两类：`GPT`和其他模型。这并非盲目吹捧，如果抛开网络和政策的限制，相信没有任何一个开发者会拒绝使用`GPT`模型去构建他们的应用程序。而在大模型应用技术开发方面，其最先新推出的`Function calling`同样带动了其他大模型在应用能力上的发展。**在`AI Agent`概念自去年引发新一轮的关注以来，`OpenAI`迅速推出了`Assistant API`，将其作为构建新一代智能代理的框架规范，用以支持`GPT`系列生态大模型的发展。**

&emsp;&emsp;`OpenAI`的`Assistant API`可实现的功能在不断的迭代，但也**因为其闭源性，很多技术细节我们无法知晓**，同时因为只可用于其自家的`GPT`系列模型，会导致使用`Assistant API`构建出的应用程序的可移植性并不高。同时对我们国内用户，还存在着网络、政策限制，导致基于它构建的应用无法进行备案，使很多小伙伴没有办法使用。但并不意味着我们要抛弃它，相反**，`OpenAI`的解决方案是非常值得我们深入研究和学习的。**

&emsp;&emsp;正如各个大模型公司在不断紧追`OpenAI`一样，我们开发者有时候虽然不能直接使用`Assistant API`，但是**通过剖析它的技术细节去学习最先进的处理思路，是可以极大程度上提升我们现有流程的效果的**。而对于个人使用，或者企业内部使用，在没有网络、备案的强制限制下，`Assistant API` 相较于其他的`AI Agent`开发框架，上手容易且效果好，还是非常建议大家使用`Assistant API`的。

&emsp;&emsp;那么`Assistant API`到底是什么？它能做到什么呢？

&emsp;&emsp;根据`OpenAI`官方的介绍**，`Assistant API` 是 `Chat Completions API` 的演变，目的是让开发人员更轻松地创建类似助手的应用。本质是降低智能应用的开发门槛。**具体来看，`Assistant`是 `OpenAI` 尝试提供给开发者的一种机制，让开发人员可以在自己的软件中以编程方式构建代理行为。助理根据指令和用户定义的任务来执行，并且可以利用模型、工具和知识来响应用户的查询。

&emsp;&emsp;从功能上看，`Assistant API`除了能提供像`Chat Completions API`一样的大模型对话交互能力，同时支持无限长的多轮对话，目前还支持三种类型的工具：**代码解释器、文件搜索和函数调用。**所谓的代码解释器，指的是能够在一个隔离的沙箱环境里编写并运行 `Python` 代码，并能够进行自我修正。而文件搜索，其实现的思路是通过自有向量数据库支持基于文件的 RAG 过程，最后对于函数调用，则是可以在各个流程中，自主决策并执行外部的自定义函数，就像我们手动实现的`ReAct Agent`一样，只不过，`OpenAI`把这个过程实现的更加强大和稳定。当然，使用`Assistant API` 仍然像`Chat Completions API` 一样是按照 token 收费的。无论多轮对话，还是RAG，所有都按实际消耗的 token 收费。这一点需要大家明确。

> 沙箱环境是一种隔离的测试环境，用于安全地运行和测试代码或应用程序，而不会对系统的其他部分造成影响。它允许开发者在控制的条件下探索和验证功能，通常用于检测漏洞和兼容性问题。

&emsp;&emsp;接下来，我们就围绕`Assistant API`的基本原理及其调用方法两个方面分别展开详细的介绍。

# 1. OpenAI Assistant API 运行原理

&emsp;&emsp;`OpenAI`发布的`Assistant API`技术的整体核心是**构建Assistant（助手）、Thread（线程）和Messages（消息）这三个概念之间的紧密关系，通过Run（运行）状态**保持与大模型的任务交互工作。

&emsp;&emsp;那么如何理解这四个抽象概念呢？ 我们来看这样一个现象级产品的例子：大家熟知的`ChatGPT`应用，每个用户的独立账号扮演的是一个“助手”的角色。在`ChatGPT`应用界面的左侧，每个单独的会话框代表一个“线程”。当点击任一历史会话，展示出来的历史对话记录则构成了“消息”。这种设计使得每次交互都既独立又连贯，便于我们追踪历史和参与实时的对话，如下图的三个阶段所示：👇

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/2501.png" width=100%></div>

&emsp;&emsp;在应用开发的逻辑中，`Assistant`可以访问`Thread`，`Thread`通过存储`Messages`并在执行对话时，自动将用户的问题与大模型的回答构建一一对应的存储机制，同时，在后续的对话中主动追加过往的历史信息，从而使大模型具备记忆能力。除此之外，Assistant API 还可以自主的根据当前大模型的最大上下文的长度，来判断是否需要对输入的文本长度进行截断，以此来保证输入信息的完整性。

&emsp;&emsp;当然，除了`Assistant`、`Thread`和`Messages`这三个类似“实体”的概念外，`Assistant API`运行机制中更加**重要的一个“动态”抽象是 `Run`（运行）状态**。`Run`的作用是将这三者有机串联，形成实际可执行的工作流程。它推动的是实际业务流程的运行。正如：“用户输入问题 - 大模型接收问题 - 大模型分析并生成回答 - 用户收到回复”，这一过程便是`Run`概念背后的实际运作方式。在 `OpenAI` 官方对 `Assistant API` 的介绍中，有这样一幅流程图用以对`Assistant API`完整生命周期的解释：👇 

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923131542441.png" width=100%></div>

&emsp;&emsp;在 `AI Agent` 概念中，每个对话都是上下文相关的。以单一`Assistant`对象为例，正如在《Ch.3 ReAct Agent 基本理论与项目实战》中构建智能客服的智能体流程，我们需要**赋予其身份背景和任务场景的设定**。所有`Thread`本质上都由同一个 `Assistant` 对象处理，基于`Assistant`对象构建出来的`Thread`，这些`Thread`用于满足不同的对话和任务需求。类似于 `ChatGPT` 应用中的每个会话列表，一旦`Thread`构建完成，不同的`Thread`是可以灵活切换的。这种灵活性得益于 `Thread` 内部对 `Messages` 的全部信息做了持久化存储。

&emsp;&emsp;**总的来说，一个 `Assistant` 可以有多个 `Thread`， 一个 `Thread` 可以有无限条 `Message`，一个用户与 `Assistant` 的多轮对话历史维护在一个 `Thread` 中。**

&emsp;&emsp;最后的`Run`是用户与任务交互的动态执行过程，既可以简单地直接调用大模型生成响应，也可以复杂到自主处理需要`Function Calling`过程的高级任务。不同 `Run` 的设定将带来完全不同的运行机制，从而满足个性化开发的需求。如下所示的顺序图（Sequence Diagram）中更加详细的描述了`Assistant API`对象之间消息交互的过程，展示了不同组件之间按时间顺序通信的具体过程：👇

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923165414686.png" width=100%></div>

&emsp;&emsp;这张顺序图具体展示了用户与Assistant API之间的交互流程，包括消息的发送和接收。主要涉及到以下几个关键步骤：

1. **创建对话线程**：
   - 用户通过发送 POST 请求到 /v1/threads 创建一个新的对话线程。(线程代表对话会话)
   - Assistant API 接收请求，并初始化一个对话线程（Thread），然后响应带有线程 ID 的信息。
<br><br>
3. **发送消息**：
   - 用户通过发送 POST 请求到 /v1/threads/{thread_id}/messages，附带消息内容，向特定的对话线程发送消息。
   - Assistant API 确认消息已收到，并处理该消息。
<br><br>
4. **消息处理与响应**：
   - 对话线程处理收到的消息，并将该消息添加到对话线程中。
   - Assistant API生成响应并存储。
<br><br>
5. **请求助手的响应**：
   - 用户通过发送 GET 请求到 /v1/threads/{thread_id}/messages 请求Assistant API的响应。
   - Assistant API 检索最新的消息并提供给用户。
<br><br>
6. **提供最后的消息**：
   - 用户请求最后的消息内容。
   - Assistant API 提供最后的消息内容给用户。

&emsp;&emsp;从上述流程大家也能看出，`Assistant API` 的内部执行过程是比较复杂的，但这些复杂的流程已由官方实现，无需我们手动去处理细节。因此，尽管**技术上复杂，但使用门槛非常低**。接下来，基于对其内部运行原理的理解，我们将按照上述流程实际操作，探讨如何通过编程快速构建一个智能助手。

# 2. Assistant API 实战入门

&emsp;&emsp;`OpenAI`官方提供了两种方法来使用 `Assistant API` 构建智能助手或代理。第一种方法是通过 [Assistants Playground](https://platform.openai.com/playground?mode=assistant) 在 OpenAI 的官网直接进行构建。第二种方法是使用 [Assistants API](https://platform.openai.com/docs/api-reference/assistants/createAssistant) 在本地编程环境中创建。这两种构建方法各有其利弊：

- **Assistants Playground（网页构建）**

&emsp;&emsp;直接通过浏览器界面操作的优势自然不用多说，在用户友好的界面上直接操作，可以即时看到结果，非常便于测试和调整助手的行为。同时无需安装额外软件，特别适合初学者或非技术用户。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923102524551.png" width=100%></div>

&emsp;&emsp;但这种方法也有明显的缺点：**它不便于集成到自动化工作流中，每次更改都需要手动操作**。因此，对于应用开发人员来说，这通常不是首选方法，所以我们这里不对这种方式展开说明，感兴趣的小伙伴可以自行尝试。而该操作页面上的各个组件的作用，在接下来关于 `Assistants API` 的介绍中，我们均会详细解释各个组件的功能、内部原理及其适用场景。

- **Assistants API（本地编程环境创建）**

&emsp;&emsp;`Assistant API`的另外一种使用方法是通过编程接口来提供服务，这就使得开发者**可以实现更复杂的功能和深度定制，同时能够轻松集成到现有的系统和工作流中，适合生产环境**。`OpenAI`官方提供了针对 `Python` 和 `Node.js` 的 SDK，以方便调用 `Assistant API` 支持的所有接口。在本系列教程中，我们将主要使用 `Python` 来进行介绍和后续的实战项目。

>  SDK（Software Development Kit）是一个软件开发包，会提供一组用于某种编程语言的工具、库和文档，使得开发人员可以更方便地创建应用程序。这个包通常封装了与特定服务或平台交互所需的API（应用程序编程接口），简化了代码的编写过程，并帮助开发者利用那些服务的特定功能。

> 例如，如果一个公司提供了一个云存储服务，它提供一个 SDK，其中包含了进行文件上传、下载、管理等操作的函数。使用 SDK，开发者可以直接调用这些函数而不需要从头编写复杂的代码来与后端系统交互。

&emsp;&emsp;接下来，我们就根据这张顺序通信流程图，具体实现 `Assistant API` 的应用过程。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923165414686.png" width=100%></div>

- **Stage 1. 创建Assistant（助手）** 

&emsp;&emsp;`Assistant`（助手）代表**一个实体，它定义的是一个智能助理或智能代理的身份背景和任务设定**，类似于培养一个专门完成某项任务的“人”。在 `AI Agent` 应用开发中，这个“人”的大脑是由大模型所替代的。因此，在创建`Assistant`对象时，我们需要在此阶段指定使用的具体模型。`Assistant`助手可以使用特定指令调用这个大模型，以完成某些具体的任务。

&emsp;&emsp;首先，创建`Assistant` 的 endpoint 是 `http://api.openai.com/v1/assistants`， 并且唯一需要必须指定的参数是 `model` ，即使用具体哪个模型。更加详细的参数请查看`OpenAI`官网👉[Create assistant](https://platform.openai.com/docs/api-reference/assistants)

> **注意：API endpoint 是专门为编程接口设计的，通常只能通过 HTTP 请求（如 POST, GET 等）来访问和操作，而不是通过普通的网页浏览器访问。需要使用编程语言（如 Python, JavaScript 等）来构建请求，并处理响应。**

&emsp;&emsp;创建 `Assistant`（助手）时，可使用的大模型基本涵盖了 `OpenAI` 的 `GPT` 系列中的整个生态，包括对话生成、图像处理、音频处理等类别。要查看可指定的具体大模型型号，第一种方法是直接访问 `OpenAI` 的官网：👉 [Available Models](https://platform.openai.com/docs/models)。

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923132417232.png" width=100%></div>

&emsp;&emsp;获取创建 `Assistant`（助手）时可用大模型的第二种方法，也是在开发过程中更常用的方法，是通过 `OpenAI` 的 `API` 接口来查询。这种方式便于在代码逻辑中灵活地处理和调用实时模型。代码如下所示：

> **请注意，这里指是 `OpenAI` 的通用 `API`，而非专指 `Assistant API`。**


```python
from openai import OpenAI

# 构建 OpenAI 客户端对象的实例
client = OpenAI()

models = client.models.list()
models
```




    SyncPage[Model](data=[Model(id='dall-e-2', created=1698798177, object='model', owned_by='system'), Model(id='gpt-4-1106-preview', created=1698957206, object='model', owned_by='system'), Model(id='tts-1-hd-1106', created=1699053533, object='model', owned_by='system'), Model(id='tts-1-hd', created=1699046015, object='model', owned_by='system'), Model(id='gpt-4-turbo-2024-04-09', created=1712601677, object='model', owned_by='system'), Model(id='gpt-4-0125-preview', created=1706037612, object='model', owned_by='system'), Model(id='gpt-4-turbo-preview', created=1706037777, object='model', owned_by='system'), Model(id='gpt-4-turbo', created=1712361441, object='model', owned_by='system'), Model(id='tts-1-1106', created=1699053241, object='model', owned_by='system'), Model(id='gpt-3.5-turbo-0613', created=1686587434, object='model', owned_by='openai'), Model(id='gpt-4o-mini', created=1721172741, object='model', owned_by='system'), Model(id='gpt-4o-mini-2024-07-18', created=1721172717, object='model', owned_by='system'), Model(id='gpt-3.5-turbo-16k-0613', created=1685474247, object='model', owned_by='openai'), Model(id='gpt-3.5-turbo-0301', created=1677649963, object='model', owned_by='openai'), Model(id='gpt-3.5-turbo-16k', created=1683758102, object='model', owned_by='openai-internal'), Model(id='text-embedding-3-small', created=1705948997, object='model', owned_by='system'), Model(id='chatgpt-4o-latest', created=1723515131, object='model', owned_by='system'), Model(id='gpt-4o-2024-08-06', created=1722814719, object='model', owned_by='system'), Model(id='gpt-3.5-turbo-1106', created=1698959748, object='model', owned_by='system'), Model(id='gpt-3.5-turbo-instruct-0914', created=1694122472, object='model', owned_by='system'), Model(id='gpt-4-0613', created=1686588896, object='model', owned_by='openai'), Model(id='gpt-3.5-turbo-0125', created=1706048358, object='model', owned_by='system'), Model(id='gpt-4', created=1687882411, object='model', owned_by='openai'), Model(id='gpt-3.5-turbo-instruct', created=1692901427, object='model', owned_by='system'), Model(id='gpt-3.5-turbo', created=1677610602, object='model', owned_by='openai'), Model(id='babbage-002', created=1692634615, object='model', owned_by='system'), Model(id='davinci-002', created=1692634301, object='model', owned_by='system'), Model(id='dall-e-3', created=1698785189, object='model', owned_by='system'), Model(id='tts-1', created=1681940951, object='model', owned_by='openai-internal'), Model(id='o1-preview', created=1725648897, object='model', owned_by='system'), Model(id='o1-preview-2024-09-12', created=1725648865, object='model', owned_by='system'), Model(id='text-embedding-ada-002', created=1671217299, object='model', owned_by='openai-internal'), Model(id='gpt-4o', created=1715367049, object='model', owned_by='system'), Model(id='gpt-4o-2024-05-13', created=1715368132, object='model', owned_by='system'), Model(id='o1-mini-2024-09-12', created=1725648979, object='model', owned_by='system'), Model(id='o1-mini', created=1725649008, object='model', owned_by='system'), Model(id='whisper-1', created=1677532384, object='model', owned_by='openai-internal'), Model(id='text-embedding-3-large', created=1705953180, object='model', owned_by='system'), Model(id='ft:gpt-3.5-turbo-0613:acmr:recipe-ner:7rOJnQow', created=1692959219, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:babbage-002:acmr::8LjwL7pi', created=1700192173, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NFcWDKd', created=1700552280, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NG1DFJy', created=1700553811, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGu7dB7', created=1700557216, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGecRZn', created=1700556254, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGoWxGp', created=1700556868, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NcxfjOE', created=1700642003, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NISonhF', created=1700563210, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIBwIXY', created=1700562164, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIJWEAY', created=1700562634, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIZytXn', created=1700563654, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:babbage-002:acmr::8ONnv6dd', created=1700822067, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PSAdThJ', created=1701077179, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Okwb8qA', created=1700911017, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Onym3eo', created=1700922684, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P3jjlnu', created=1700983255, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P7Wp9ZW', created=1700997831, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8ICr9h', created=1701000768, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8JcJOS', created=1701000856, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8MZq4B', created=1701001039, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8eJWFu', created=1701002139, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8jDYPa', created=1701002444, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8v5J5I', created=1701003179, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PjVLXnJ', created=1701143811, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PjglJDQ', created=1701144519, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmHw3M0', created=1701154513, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmrWm2Q', created=1701156718, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pmt7LeV', created=1701156817, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmkRMhD', created=1701156280, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pmrraox', created=1701156739, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmsqhY1', created=1701156800, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmvMoSo', created=1701156956, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pn0GIjR', created=1701157261, object='model', owned_by='acmr-wtr5tv'), Model(id='ft:gpt-3.5-turbo-1106:acmr::8Q8YIAz8', created=1701240094, object='model', owned_by='acmr-wtr5tv')], object='list')



&emsp;&emsp;通过`.models.list()` 方法可查询所有可用的模型，返回的是一个包含多个模型详细信息的列表：


```python
models.data
```




    [Model(id='dall-e-2', created=1698798177, object='model', owned_by='system'),
     Model(id='gpt-4-1106-preview', created=1698957206, object='model', owned_by='system'),
     Model(id='tts-1-hd-1106', created=1699053533, object='model', owned_by='system'),
     Model(id='tts-1-hd', created=1699046015, object='model', owned_by='system'),
     Model(id='gpt-4-turbo-2024-04-09', created=1712601677, object='model', owned_by='system'),
     Model(id='gpt-4-0125-preview', created=1706037612, object='model', owned_by='system'),
     Model(id='gpt-4-turbo-preview', created=1706037777, object='model', owned_by='system'),
     Model(id='gpt-4-turbo', created=1712361441, object='model', owned_by='system'),
     Model(id='tts-1-1106', created=1699053241, object='model', owned_by='system'),
     Model(id='gpt-3.5-turbo-0613', created=1686587434, object='model', owned_by='openai'),
     Model(id='gpt-4o-mini', created=1721172741, object='model', owned_by='system'),
     Model(id='gpt-4o-mini-2024-07-18', created=1721172717, object='model', owned_by='system'),
     Model(id='gpt-3.5-turbo-16k-0613', created=1685474247, object='model', owned_by='openai'),
     Model(id='gpt-3.5-turbo-0301', created=1677649963, object='model', owned_by='openai'),
     Model(id='gpt-3.5-turbo-16k', created=1683758102, object='model', owned_by='openai-internal'),
     Model(id='text-embedding-3-small', created=1705948997, object='model', owned_by='system'),
     Model(id='chatgpt-4o-latest', created=1723515131, object='model', owned_by='system'),
     Model(id='gpt-4o-2024-08-06', created=1722814719, object='model', owned_by='system'),
     Model(id='gpt-3.5-turbo-1106', created=1698959748, object='model', owned_by='system'),
     Model(id='gpt-3.5-turbo-instruct-0914', created=1694122472, object='model', owned_by='system'),
     Model(id='gpt-4-0613', created=1686588896, object='model', owned_by='openai'),
     Model(id='gpt-3.5-turbo-0125', created=1706048358, object='model', owned_by='system'),
     Model(id='gpt-4', created=1687882411, object='model', owned_by='openai'),
     Model(id='gpt-3.5-turbo-instruct', created=1692901427, object='model', owned_by='system'),
     Model(id='gpt-3.5-turbo', created=1677610602, object='model', owned_by='openai'),
     Model(id='babbage-002', created=1692634615, object='model', owned_by='system'),
     Model(id='davinci-002', created=1692634301, object='model', owned_by='system'),
     Model(id='dall-e-3', created=1698785189, object='model', owned_by='system'),
     Model(id='tts-1', created=1681940951, object='model', owned_by='openai-internal'),
     Model(id='o1-preview', created=1725648897, object='model', owned_by='system'),
     Model(id='o1-preview-2024-09-12', created=1725648865, object='model', owned_by='system'),
     Model(id='text-embedding-ada-002', created=1671217299, object='model', owned_by='openai-internal'),
     Model(id='gpt-4o', created=1715367049, object='model', owned_by='system'),
     Model(id='gpt-4o-2024-05-13', created=1715368132, object='model', owned_by='system'),
     Model(id='o1-mini-2024-09-12', created=1725648979, object='model', owned_by='system'),
     Model(id='o1-mini', created=1725649008, object='model', owned_by='system'),
     Model(id='whisper-1', created=1677532384, object='model', owned_by='openai-internal'),
     Model(id='text-embedding-3-large', created=1705953180, object='model', owned_by='system'),
     Model(id='ft:gpt-3.5-turbo-0613:acmr:recipe-ner:7rOJnQow', created=1692959219, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:babbage-002:acmr::8LjwL7pi', created=1700192173, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NFcWDKd', created=1700552280, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NG1DFJy', created=1700553811, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGu7dB7', created=1700557216, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGecRZn', created=1700556254, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NGoWxGp', created=1700556868, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NcxfjOE', created=1700642003, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NISonhF', created=1700563210, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIBwIXY', created=1700562164, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIJWEAY', created=1700562634, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8NIZytXn', created=1700563654, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:babbage-002:acmr::8ONnv6dd', created=1700822067, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PSAdThJ', created=1701077179, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Okwb8qA', created=1700911017, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Onym3eo', created=1700922684, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P3jjlnu', created=1700983255, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P7Wp9ZW', created=1700997831, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8ICr9h', created=1701000768, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8JcJOS', created=1701000856, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8MZq4B', created=1701001039, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8eJWFu', created=1701002139, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8jDYPa', created=1701002444, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8P8v5J5I', created=1701003179, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PjVLXnJ', created=1701143811, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PjglJDQ', created=1701144519, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmHw3M0', created=1701154513, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmrWm2Q', created=1701156718, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pmt7LeV', created=1701156817, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmkRMhD', created=1701156280, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pmrraox', created=1701156739, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmsqhY1', created=1701156800, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8PmvMoSo', created=1701156956, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Pn0GIjR', created=1701157261, object='model', owned_by='acmr-wtr5tv'),
     Model(id='ft:gpt-3.5-turbo-1106:acmr::8Q8YIAz8', created=1701240094, object='model', owned_by='acmr-wtr5tv')]



&emsp;&emsp;可以通过索引提取具体的某一个模型的详细信息。


```python
models.data[0]
```




    Model(id='dall-e-2', created=1698798177, object='model', owned_by='system')



&emsp;&emsp;具体模型的名称存储在返回消息列表中的`id`键对应的值中，所以要获取到每个具体的模型名称，可以通过遍历数组中的每个模型对象，然后从每个对象中提取并打印 `id` 字段。代码如下所示：


```python
# 遍历模型列表并打印每个模型的 ID
for model in models.data:
    print(model.id)
```

    dall-e-2
    gpt-4-1106-preview
    tts-1-hd-1106
    tts-1-hd
    gpt-4-turbo-2024-04-09
    gpt-4-0125-preview
    gpt-4-turbo-preview
    gpt-4-turbo
    tts-1-1106
    gpt-3.5-turbo-0613
    gpt-4o-mini
    gpt-4o-mini-2024-07-18
    gpt-3.5-turbo-16k-0613
    gpt-3.5-turbo-0301
    gpt-3.5-turbo-16k
    text-embedding-3-small
    chatgpt-4o-latest
    gpt-4o-2024-08-06
    gpt-3.5-turbo-1106
    gpt-3.5-turbo-instruct-0914
    gpt-4-0613
    gpt-3.5-turbo-0125
    gpt-4
    gpt-3.5-turbo-instruct
    gpt-3.5-turbo
    babbage-002
    davinci-002
    dall-e-3
    tts-1
    o1-preview
    o1-preview-2024-09-12
    text-embedding-ada-002
    gpt-4o
    gpt-4o-2024-05-13
    o1-mini-2024-09-12
    o1-mini
    whisper-1
    text-embedding-3-large
    ft:gpt-3.5-turbo-0613:acmr:recipe-ner:7rOJnQow
    ft:babbage-002:acmr::8LjwL7pi
    ft:gpt-3.5-turbo-1106:acmr::8NFcWDKd
    ft:gpt-3.5-turbo-1106:acmr::8NG1DFJy
    ft:gpt-3.5-turbo-1106:acmr::8NGu7dB7
    ft:gpt-3.5-turbo-1106:acmr::8NGecRZn
    ft:gpt-3.5-turbo-1106:acmr::8NGoWxGp
    ft:gpt-3.5-turbo-1106:acmr::8NcxfjOE
    ft:gpt-3.5-turbo-1106:acmr::8NISonhF
    ft:gpt-3.5-turbo-1106:acmr::8NIBwIXY
    ft:gpt-3.5-turbo-1106:acmr::8NIJWEAY
    ft:gpt-3.5-turbo-1106:acmr::8NIZytXn
    ft:babbage-002:acmr::8ONnv6dd
    ft:gpt-3.5-turbo-1106:acmr::8PSAdThJ
    ft:gpt-3.5-turbo-1106:acmr::8Okwb8qA
    ft:gpt-3.5-turbo-1106:acmr::8Onym3eo
    ft:gpt-3.5-turbo-1106:acmr::8P3jjlnu
    ft:gpt-3.5-turbo-1106:acmr::8P7Wp9ZW
    ft:gpt-3.5-turbo-1106:acmr::8P8ICr9h
    ft:gpt-3.5-turbo-1106:acmr::8P8JcJOS
    ft:gpt-3.5-turbo-1106:acmr::8P8MZq4B
    ft:gpt-3.5-turbo-1106:acmr::8P8eJWFu
    ft:gpt-3.5-turbo-1106:acmr::8P8jDYPa
    ft:gpt-3.5-turbo-1106:acmr::8P8v5J5I
    ft:gpt-3.5-turbo-1106:acmr::8PjVLXnJ
    ft:gpt-3.5-turbo-1106:acmr::8PjglJDQ
    ft:gpt-3.5-turbo-1106:acmr::8PmHw3M0
    ft:gpt-3.5-turbo-1106:acmr::8PmrWm2Q
    ft:gpt-3.5-turbo-1106:acmr::8Pmt7LeV
    ft:gpt-3.5-turbo-1106:acmr::8PmkRMhD
    ft:gpt-3.5-turbo-1106:acmr::8Pmrraox
    ft:gpt-3.5-turbo-1106:acmr::8PmsqhY1
    ft:gpt-3.5-turbo-1106:acmr::8PmvMoSo
    ft:gpt-3.5-turbo-1106:acmr::8Pn0GIjR
    ft:gpt-3.5-turbo-1106:acmr::8Q8YIAz8


&emsp;&emsp;这里得到的模型名称，如`ft:gpt-3.5-turbo-1106:acmr:recipe-ner:7rOJnQow` 和类似的，表示的是 OpenAI 的微调模型（Fine-tuned Models），大家的私有账号没有是正常的，我们关注和使用的是由`OpenAI`提供的通用大模型。

&emsp;&emsp;**需要明确的是：`Assistant`对象由 `OpenAI API` 的客户创建**，所以其使用的`API Key`依然是 `OpenAI` 账户中的 `API Key`，不需要为`Assistant API`单独创建。

&emsp;&emsp;一个最简单的`Assistant`（助理）在创建时只需要传入`model`参数，用以指定使用哪个具体的大模型。这里我们先讨论其基本参数（注意：`Assistant API` endpoint并非仅如下几个参数，其他自定义参数我们将在接下来的进阶使用部分再详细介绍），如下所示： 👇

> Create assistant ：https://platform.openai.com/docs/api-reference/assistants

| 字段名称    | 类型           | 是否必需  | 描述                                                  | 最大长度      |
|---------|--------------|-------|-----------------------------------------------------|-----------|
| model   | string       | 必需    | 使用的模型 ID | 无限制       |
| name    | string/null  | 可选    | 助理的身份/名称                                              | 256 字符     |
| instructions | string/null  | 可选    | 助理的任务/目标描述（系统消息）                                        | 256,000 字符 |

&emsp;&emsp;在介绍`ReAct Agent`的课程中我们就一直强调，**给`AI Agent`设定明确的身份和目标对其执行任务的效果有显著影响**，这一理论对`Assistant API`也同样适用。因此，在实际使用时，除了`model`参数外，我们建议将`name`（名称）和`instructions`（指令）参数也视为必填项，以确保智能助手能够有效地完成其任务。因此，具体构建的代码如下所示：


```python
from openai import OpenAI

# 构建 OpenAI 客户端对象的实例
client = OpenAI()

assistant = client.beta.assistants.create(
    model="gpt-4o-mini-2024-07-18",
    name="Good writer",  # 优秀的作家
    instructions="You are an expert at writing excellent literature"  # 你是一位善于写优秀文学作品的专家
)
```

&emsp;&emsp; `Assistants API` 需要传递 `beta` 的 HTTP Header，即：OpenAI-Beta: assistants=v2 ， 但**对于使用官方提供的 `Python` 或者 `Node.js` SDK来说，会自动处理这个问题。**所以我们在构建的时候可以直接使用`client.beta.assistants.create` 方法创建一个新的`Assistant`对象实例，通过`name` 和 `instructions` 参数将其设置为专门用于写作的智能助手。其返回的响应体如下所示：


```python
assistant
```




    Assistant(id='asst_DQ41JJBwmGwhx8LOp6u76lST', created_at=1727257878, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0)



&emsp;&emsp;可以通过`to_dict()`方法将 `assistant` 对象的所有属性（如模型名称、指定的名称和指令等）转换成一个易于阅读和处理的形式。如下所示：


```python
assistant.to_dict()
```




    {'id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
     'created_at': 1727257878,
     'description': None,
     'instructions': 'You are an expert at writing excellent literature',
     'metadata': {},
     'model': 'gpt-4o-mini-2024-07-18',
     'name': 'Good writer',
     'object': 'assistant',
     'tools': [],
     'response_format': 'auto',
     'temperature': 1.0,
     'tool_resources': {},
     'top_p': 1.0}



&emsp;&emsp;在上述返回的响应体中我们先关注键值不为空的有效返回值的对象类型：

- **id**：实体的唯一标识，一个 API_KEY 可以创建多个 Assistant 实例。
- **created_at**：创建该助理的 Unix 时间戳（秒）。
- **instructions**：助理使用的系统指令，指导助手的个性并定义其目标。
- **model**：使用的模型 ID。
- **name**：助理的名称，最大长度为 256 个字符。
- **object**：对象类型，始终是助理（assistant）。
- **temperature、top_p**：大模型的生成参数。

&emsp;&emsp;**生成`Assistant`（助手）实例后，其不会产生任何对话等行为**，我们需要按照顺序通信流程图继续构建流程：基于该`Assistant`实例，创建其对应的`Thread`（线程）。

- **Stage 2. 创建Thread（线程）**

&emsp;&emsp;在前面的内容中我们已经解释过，`Thread`（线程）即会话，每个`Assistant`对象可以应用多个不同的`Thread`，用来执行多个不同的聊天或者任务。但是**需要注意的是：`Thread` 实例与 `Assistant` 实例并不是直接绑定的，也就是说：我们可以分别单独的创建`Assistant`对象和`Thread`对象，并不需要在创建阶段建立绑定关系，只需要在执行会话/任务时将其绑定即可。**

&emsp;&emsp;基于这种机制，创建`Thread`的一种最简单的方式是无需任何参数，直接调用`https://api.openai.com/v1/threads`接口即可。`Assistant API`抽象出的接口调用方法如下所示：

> https://platform.openai.com/docs/api-reference/threads/createThread


```python
thread = client.beta.threads.create()
```

&emsp;&emsp; 通过 `OpenAI`客户端的实例，直接调用其`.beta.threads.create` 方法创建一个新的对话线程。同样可以使用`to_dict()`格式化返回的响应体内容展示：


```python
thread.to_dict()
```




    {'id': 'thread_phxLhtF42ZaswePBms3vVbKf',
     'created_at': 1727258023,
     'metadata': {},
     'object': 'thread',
     'tool_resources': {}}



&emsp;&emsp;创建`Thread`实例对象的返回响应体内容就比较简单，通过`id`来做对话线程的唯一标识符，同时用`created_at`和`object`字段分别标识`Thread`创建的时间和对象类型。

&emsp;&emsp;接下来，我们继续构建`Message`对象过程。

- **Stage 3. 向`Thread`（线程）中添加`Message`（消息）**

&emsp;&emsp;`Message`指的是`Assistant`和用户之间的对话会话，但所有的会话信息，都存放在`Thread`中。也就是说：**当在具有用户上下文的线程上添加第一条消息时，它会添加到该线程上。当运行该线程时，助手会将其消息响应作为响应添加到同一线程上。** 其过程如下图所示：👇

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/2502.png" width=80%></div>

&emsp;&emsp;当触发对话操作时，`Assistant API`的机制会实时的将用户的消息添加到该线程中，这样可以保留对话的状态。所以当我们使用`Assistant`（助手）的时候，一个好处是**它可以自主进行对话管理，开发者不需要手动处理所产生的聊天历史记录**。**消息可以包括文本、图像和其他文件，每个线程的消息数限制为 100,000 条。**当超出这个限制后，会触发 `Assistant API` 的一种策略自动执行截断操作（截断策略稍后进行详细介绍）。

> Messages：https://platform.openai.com/docs/api-reference/messages

&emsp;&emsp;`Messages`的endpoint 为：`https://api.openai.com/v1/threads/{thread_id}/messages`， 该接口核心参数如下：

| 字段名     | 类型           | 是否必需 | 描述                                                           |
|----------|--------------|--------|--------------------------------------------------------------|
| thread_id | string       | 必需     | 用于创建消息的线程 ID。                                             |
| role      | string       | 必需     | 创建消息的实体的角色。允许的值包括：<br>user：表示消息由实际用户发送。<br>assistant：表示消息由助手生成。 |
| content   | string/array | 必需     | 消息的内容。                                                     |


&emsp;&emsp;正如上述所说，`Assistant API`可以保留会话的状态，会话的状态依赖于历史的会话信息，而历史的会话信息又全部都存放在`Thread`中，因此，在生成`Messages`时必须传递用于将其存储`Thread id`，代码如下：


```python
thread.id
```




    'thread_phxLhtF42ZaswePBms3vVbKf'




```python
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动"
)
```

&emsp;&emsp;这种构建对话的方法非常类似于`chat.completions` 调用方式。在指定 `Thread id` 的同时，通过 `role` 和 `content` 字段传递对话内容，明确告诉大模型这是用户提出的问题。


```python
message.to_dict()
```




    {'id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
     'assistant_id': None,
     'attachments': [],
     'content': [{'text': {'annotations': [],
        'value': '写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动'},
       'type': 'text'}],
     'created_at': 1727258147,
     'metadata': {},
     'object': 'thread.message',
     'role': 'user',
     'run_id': None,
     'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'}



&emsp;&emsp;同样，这里我们先关注有实际键值的参数，如下所示：

- **id**：消息对象的唯一标识。
- **content**: 数组，消息内容，包括文本
- **object**: 字符串类型，对象类型始终是 thread.message。
- **created_at**: 整型，消息创建时的 Unix 时间戳（秒）。
- **role**: 消息的角色标识，指出当前 context 中的 value，是用户的提问
- **thread_id**: 字符串类型，此消息所属的线程 ID。

&emsp;&emsp;需要重点说明的是：上述这个过程只是把用户此轮提出的问题追加的`Thread`内部维护的消息列表中，还并未实际的去调用大模型生成响应。

- **Stage 4. 创建Run（运行）**

&emsp;&emsp;如果**要让 `Assistant` 响应 `Thread` 中最新追加进来的消息，在`Assistant API` 运行机制中需要创建一个 `Run` (运行)状态，指示 `Assistant` 读取 `Thread` 中的消息并采取适当的操作。**

&emsp;&emsp;具体来看，`Run`（运行）状态，指的是将`Thread`中最新添加进来的`Messages`信息，发送给`Assistant`（助理）对象实例，这个`Assistant`对象实例中指定了使用哪个模型、它的身份是什么，它的任务目标是什么，在这样的设定下，去回答用户此轮提出的问题。所以其必须传递的参数如下所示：👇

> https://platform.openai.com/docs/api-reference/runs/createRun

| 字段名          | 类型          | 是否必需 | 描述                                                                                                    |
|---------------|-------------|--------|-------------------------------------------------------------------------------------------------------|
| thread_id     | string      | 必需     | 用于运行的线程 ID。                                                                                          |
| assistant_id  | string      | 必需     | 助手的 ID。                                                                                            |


&emsp;&emsp;如下代码所示，通过`threads.runs.create()`方法在特定对话线程中启动一个新的运行状态：


```python
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
```


```python
run.to_dict()
```




    {'id': 'run_iH1Ll2mLI0SdARhAEuec4tz4',
     'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
     'cancelled_at': None,
     'completed_at': None,
     'created_at': 1727258230,
     'expires_at': 1727258830,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert at writing excellent literature',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o-mini-2024-07-18',
     'object': 'thread.run',
     'parallel_tool_calls': True,
     'required_action': None,
     'response_format': 'auto',
     'started_at': None,
     'status': 'queued',
     'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': None,
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}



&emsp;&emsp;我们关注有实际键值的参数，如下所示：

- **id**: 字符串。运行状态的唯一标识
- **assistant_id**: 字符串。用于执行此运行的助手 ID。
- **created_at**: 整数。创建运行时的 Unix 时间戳（秒）。
- **expires_at**:运行结束时的Unix时间戳(以秒为单位)。
- **instructions**: 字符串。助理使用的系统指令，指导助手的个性并定义其目标。
- **model**: 字符串。助手在此运行中使用的模型。
- **object**: 字符串。对象类型，总是 thread.run。  
- **thread_id**: 字符串。执行运行的线程 ID。
- **status**: 字符串。运行的状态，可能为 queued（排队中）、in_progress（进行中）、requires_action（需要操作）、cancelling（取消中）、cancelled（已取消）、failed（失败）、completed（已完成）、incomplete（不完整）或 expired（已过期）。

&emsp;&emsp;在`Run`状态下返回的响应体数据中能明显发现会复杂很多，而**实际上 `Run` 的底层是一个异步调用的过程**，意味着它不会等大模型响应完所有待处理请求再返回，而是直接返回响应结果。这里最值得关注的是`status`字段，我们来详细的看一下`Run`的运行原理。如下图所示：👇

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/2503.png" width=80%></div>

&emsp;&emsp;如上流程图所示，**在`Run`的处理机制中是有状态的，不同状态之间会根据实时的处理情况进行转移**。在线程上调用助手，助手使用其配置和线程的消息通过调用模型和工具来执行任务。作为运行的一部分，助手将最终的响应作为assistant消息添加到线程中，以此完成一次完整的运行生命周期。

&emsp;&emsp;在理解了`Run`在运行状态时的原理后，我们先来看一下，当想要检索并查看在当前线程中最新的大模型的响应消息时，应该如何操作？

&emsp;&emsp;首先，必须通过`role`（角色）来识别具体的消息，对于大模型的响应回复来说，其键值一定是`role`:`assistant`。


```python
response = client.beta.threads.messages.list(thread_id=thread.id)
```

&emsp;&emsp;通过`.beta.threads.messages.list`方法可以指定对话线程的 ID，获取其历史交互的所有消息记录。


```python
response
```




    SyncCursorPage[Message](data=[Message(id='msg_RidBVSlednYxDJb83UpVehqH', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='在一个阳光明媚的早晨，小女孩小玲决定走出家门，去附近的森林探险。她背上一个小背包，里面装着几块饼干和一瓶水，兴奋地踏上了通往森林的小道。一路上，鸟儿在树梢上欢快地鸣唱，花儿在微风中轻轻摇摆，整个世界都显得那么生机勃勃。\n\n走进森林后，阳光透过树叶洒下斑驳的光影，小玲感到了一丝神秘的气息。她不知道自己在森林里究竟会遇到什么，心里既期待又有些紧张。小玲喜欢冒险，然而未知的事物终究会让她心头泛起一丝怯意。\n\n就在她走得深入的时候，突然听见了一阵低沉的咕噜声。那声音像是在远处的某个地方回响，似乎带着一种奇异的韵律。小玲停下脚步，四处张望，试图找出声源。她的心跳开始加速，既想靠近去看个究竟，又有些害怕。\n\n她鼓起勇气，缓步走向声音传来的方向。越走越近，咕噜声渐渐清晰，夹杂着偶尔的树枝折断声。小玲的好奇心战胜了心中的恐惧，终于走到了一个开阔的空地上。然而，她目睹的景象令她大吃一惊：在空地中央，矗立着一只巨大的怪兽。\n\n这只怪兽浑身长满了色彩斑斓的鳞片，犹如大海中的珊瑚，反射出五光十色的光芒。它的眼睛像两个闪烁的宝石，注视着小玲，嘴巴张开时露出的长牙如同白色的柱子，显得既可怕又神秘。小玲的心中充满了恐惧和疑惑：“这是个什么生物？它会不会攻击我？”\n\n可是，出乎意料的是，怪兽似乎并不打算伤害她。它只是在空地上悠闲地活动，偶尔低下头来，用长长的舌头舔一舔地上的草丛。小玲开始感到一丝奇特的安慰：“它并不凶恶，也许我可以和它交朋友。”\n\n鼓起勇气，小玲一步一步走近怪兽。她轻声说道：“嗨，你好，你的名字是什么？”怪兽听到小玲的声音，转过头来，眼中流露出一种温暖的光芒。它似乎理解了小玲的话，慢慢地向她靠近，发出了一阵柔和低吟的声音，像是在回应她的问候。\n\n小玲心中涌起一阵欢喜，她渐渐忘记了刚才的恐惧。她慢慢伸出手，试探性地抚摸怪兽的鳞片，虽然有些粗糙，但却暖和如同阳光。怪兽用头轻轻蹭了蹭小玲的手，仿佛在邀请她一起玩耍。\n\n在那片森林里，小玲和怪兽像朋友一样快乐地嬉戏，草地上留下了她们的笑声和欢呼。时间在不知不觉中流逝，小玲深深体会到，每一个看似可怕的事物，背后都可能隐藏着温柔和友好。\n\n当夕阳西下，小玲知道是时候回家了。和怪兽依依惜别时，她心中不禁涌起一阵留恋：“我一定会再来的，你要等我哦！”怪兽轻轻点了点头，仿佛在回应着她。\n\n回家的路上，小玲的心情愉快无比，她暗下决心，明天还要再来这片森林，去寻找那个有趣的朋友。她明白了，勇气和开放的心灵，才能让她遇见世界上最美好的事物。'), type='text')], created_at=1727258232, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_iH1Ll2mLI0SdARhAEuec4tz4', status=None, thread_id='thread_phxLhtF42ZaswePBms3vVbKf'), Message(id='msg_6lm5pzwVIfPemH6uBZAshsGZ', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动'), type='text')], created_at=1727258147, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_phxLhtF42ZaswePBms3vVbKf')], object='list', first_id='msg_RidBVSlednYxDJb83UpVehqH', last_id='msg_6lm5pzwVIfPemH6uBZAshsGZ', has_more=False)




```python
response.data[0].content[0].text.value
```




    '在一个阳光明媚的早晨，小女孩小玲决定走出家门，去附近的森林探险。她背上一个小背包，里面装着几块饼干和一瓶水，兴奋地踏上了通往森林的小道。一路上，鸟儿在树梢上欢快地鸣唱，花儿在微风中轻轻摇摆，整个世界都显得那么生机勃勃。\n\n走进森林后，阳光透过树叶洒下斑驳的光影，小玲感到了一丝神秘的气息。她不知道自己在森林里究竟会遇到什么，心里既期待又有些紧张。小玲喜欢冒险，然而未知的事物终究会让她心头泛起一丝怯意。\n\n就在她走得深入的时候，突然听见了一阵低沉的咕噜声。那声音像是在远处的某个地方回响，似乎带着一种奇异的韵律。小玲停下脚步，四处张望，试图找出声源。她的心跳开始加速，既想靠近去看个究竟，又有些害怕。\n\n她鼓起勇气，缓步走向声音传来的方向。越走越近，咕噜声渐渐清晰，夹杂着偶尔的树枝折断声。小玲的好奇心战胜了心中的恐惧，终于走到了一个开阔的空地上。然而，她目睹的景象令她大吃一惊：在空地中央，矗立着一只巨大的怪兽。\n\n这只怪兽浑身长满了色彩斑斓的鳞片，犹如大海中的珊瑚，反射出五光十色的光芒。它的眼睛像两个闪烁的宝石，注视着小玲，嘴巴张开时露出的长牙如同白色的柱子，显得既可怕又神秘。小玲的心中充满了恐惧和疑惑：“这是个什么生物？它会不会攻击我？”\n\n可是，出乎意料的是，怪兽似乎并不打算伤害她。它只是在空地上悠闲地活动，偶尔低下头来，用长长的舌头舔一舔地上的草丛。小玲开始感到一丝奇特的安慰：“它并不凶恶，也许我可以和它交朋友。”\n\n鼓起勇气，小玲一步一步走近怪兽。她轻声说道：“嗨，你好，你的名字是什么？”怪兽听到小玲的声音，转过头来，眼中流露出一种温暖的光芒。它似乎理解了小玲的话，慢慢地向她靠近，发出了一阵柔和低吟的声音，像是在回应她的问候。\n\n小玲心中涌起一阵欢喜，她渐渐忘记了刚才的恐惧。她慢慢伸出手，试探性地抚摸怪兽的鳞片，虽然有些粗糙，但却暖和如同阳光。怪兽用头轻轻蹭了蹭小玲的手，仿佛在邀请她一起玩耍。\n\n在那片森林里，小玲和怪兽像朋友一样快乐地嬉戏，草地上留下了她们的笑声和欢呼。时间在不知不觉中流逝，小玲深深体会到，每一个看似可怕的事物，背后都可能隐藏着温柔和友好。\n\n当夕阳西下，小玲知道是时候回家了。和怪兽依依惜别时，她心中不禁涌起一阵留恋：“我一定会再来的，你要等我哦！”怪兽轻轻点了点头，仿佛在回应着她。\n\n回家的路上，小玲的心情愉快无比，她暗下决心，明天还要再来这片森林，去寻找那个有趣的朋友。她明白了，勇气和开放的心灵，才能让她遇见世界上最美好的事物。'



&emsp;&emsp;随着对话次数的增加，使用`.beta.threads.messages.list`方法返回的消息列表就越多。这里我们继续进行两轮对话测试：


```python
message_2 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇关于孙悟空大闹天宫的精彩战斗故事"
)
```


```python
run_2 = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
```


```python
run_2.to_dict()
```




    {'id': 'run_sSvOaN2szuGCHYS8AmTEtLHW',
     'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
     'cancelled_at': None,
     'completed_at': None,
     'created_at': 1727258511,
     'expires_at': 1727259111,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert at writing excellent literature',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o-mini-2024-07-18',
     'object': 'thread.run',
     'parallel_tool_calls': True,
     'required_action': None,
     'response_format': 'auto',
     'started_at': None,
     'status': 'queued',
     'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': None,
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}




```python
response_2 = client.beta.threads.messages.list(thread_id=thread.id)
```


```python
response_2.to_dict()
```




    {'data': [{'id': 'msg_kIG2Cxn2lqJ42dGYmLUdNEen',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在遥远的古代，天宫中的神仙们过着安详而宁静的生活。然而，一场动乱悄然降临，因为孙悟空，这位石猴，刚刚取得了无比强大的法力，决意要在天宫中大展宏图。一切就此拉开序幕，接下来他将以不可一世的姿态，翻天覆地。\n\n那一天，阳光明媚，天宫的众神正在举行盛大的庆典，庆祝天帝的长寿。突然，天宫外传来了震耳欲聋的吼声，随之而来的，是一个身影如电般飞驰而来，正是孙悟空。他化作一阵狂风，瞬间闯入了天宫的殿堂。\n\n“谁敢拦我！”他的声音如雷霆，震动着整个天宫。众仙一阵惊慌，纷纷朝四周奔逃。天兵天将们立刻反应过来，拿起武器，准备迎战。然而，面对这个曾经在花果山狂妄称王的石猴，他们内心的不安油然而生。\n\n孙悟空高高跃起，手中握着他的如意金箍棒，顺手挥动，金光闪烁，仿佛要撕裂天空。他的身形迅速变换，转瞬之间，变成了无数个猴影，让天兵天将无法捕捉到他的真实位置。 “看招！”他一声怒吼，金箍棒瞬间化作十倍的长度，狠狠地朝最近的天将砸去。\n\n“挡住他！”天将们纷纷聚拢，用力架住武器。然而，他们却只能听见“轰”的一声巨响，伴随着震耳欲聋的冲击波，天将们纷纷被击退，狼狈不堪。孙悟空趁机纵身而起，他的武技如风，任意游走于战场之上，天兵天将们根本无法抵挡。\n\n“碧空之上，谁敢与我一战？”孙悟空大喊，眼中燃烧着不屈的怒火。就在此时，天宫中的众神纷纷派出强大的儒雅仙神，试图拦住孙悟空的正义之路。那位名叫卷帘大将的神将，手持紫金钺，直抵云霄，气势攀升，蓦地冲向孙悟空。\n\n“你也想来试试？”孙悟空嘴角勾起一抹轻蔑的笑容，随即一跃而起，他的金箍棒再度变幻，直逼卷帘大将。两者在空中交锋，气劲如虹，斗技令人屏息以待。卷帘大将的钺与孙悟空的棒，在空中不断擦撞，四周的云彩被震得破碎，瞬间化为无数光点！\n\n战斗愈演愈烈，直到最终，卷帘大将终于被孙悟空的力道压制，他痛苦地跌落在地。紧接着，众天兵天将接踵而至，然而孙悟空已然势不可挡。他以无与伦比的速度和力量，扫荡整个天宫，众神惊吓退避，神仙也无暇应对。\n\n然而，天宫中的如来佛祖此时也注意到了这一切，他暗自思量，决定出手制止这场混乱。佛祖轻轻一抬手，蓦然间，整个天宫的气氛骤然一变。 shimmering 慢慢亮起，一道道金光闪烁，直逼孙悟空而来。孙悟空心中一凛，他意识到这位佛祖的威势非同小可，心中既惊又怒，“我倒要看看你有什么本事能够压住我！”\n\n孙悟空挺身而出，与如来佛祖展开了激烈的交手。他挥舞金箍棒，施展出百般法术，然而如来佛祖依然从容应对，仿佛这场战斗如行云流水，游刃有余。最终，孙悟空的力量被如来佛祖轻易化解，“你果然是强大的法力，但此处乃天宫，岂容你这般放肆！”\n\n一声喝止，孙悟空瞬间感受到一股无形的压力，犹如山岳压顶。他意识到自己在前力量上遇到了瓶颈，心中稍微不甘，但眼见眼前的如来佛祖，诸多感情蜂拥而至。“你为何要阻止我？”他问，眼中透露出不屈的挑战。\n\n“世间无恶，何苦与天斗？”如来佛祖温文尔雅，但其中的坚定和威严让孙悟空不得不动摇。经过一番斗智斗勇，最后孙悟空终于意识到，继续这样的战斗只能徒劳，反而会让自己陷入更深的绝境。\n\n“好吧，我认输，不过我一定会回来的！”孙悟空看着如来佛祖，消散了心中的怒火，松开了金箍棒，默默退回了自己的花果山。\n\n从此，他的心中留下一道无法磨灭的烙印，天宫虽好，但他明白自己的路还远未结束，未来还将有更多的冒险与挑战。大闹天宫，孙悟空不仅是一次蛮横的反抗，更是他追寻自我的一段传奇史诗。'},
         'type': 'text'}],
       'created_at': 1727258513,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_sSvOaN2szuGCHYS8AmTEtLHW',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_mcVj20d1FQDLX69oHQ2Ci9sH',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '写一篇关于孙悟空大闹天宫的精彩战斗故事'},
         'type': 'text'}],
       'created_at': 1727258509,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_RidBVSlednYxDJb83UpVehqH',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在一个阳光明媚的早晨，小女孩小玲决定走出家门，去附近的森林探险。她背上一个小背包，里面装着几块饼干和一瓶水，兴奋地踏上了通往森林的小道。一路上，鸟儿在树梢上欢快地鸣唱，花儿在微风中轻轻摇摆，整个世界都显得那么生机勃勃。\n\n走进森林后，阳光透过树叶洒下斑驳的光影，小玲感到了一丝神秘的气息。她不知道自己在森林里究竟会遇到什么，心里既期待又有些紧张。小玲喜欢冒险，然而未知的事物终究会让她心头泛起一丝怯意。\n\n就在她走得深入的时候，突然听见了一阵低沉的咕噜声。那声音像是在远处的某个地方回响，似乎带着一种奇异的韵律。小玲停下脚步，四处张望，试图找出声源。她的心跳开始加速，既想靠近去看个究竟，又有些害怕。\n\n她鼓起勇气，缓步走向声音传来的方向。越走越近，咕噜声渐渐清晰，夹杂着偶尔的树枝折断声。小玲的好奇心战胜了心中的恐惧，终于走到了一个开阔的空地上。然而，她目睹的景象令她大吃一惊：在空地中央，矗立着一只巨大的怪兽。\n\n这只怪兽浑身长满了色彩斑斓的鳞片，犹如大海中的珊瑚，反射出五光十色的光芒。它的眼睛像两个闪烁的宝石，注视着小玲，嘴巴张开时露出的长牙如同白色的柱子，显得既可怕又神秘。小玲的心中充满了恐惧和疑惑：“这是个什么生物？它会不会攻击我？”\n\n可是，出乎意料的是，怪兽似乎并不打算伤害她。它只是在空地上悠闲地活动，偶尔低下头来，用长长的舌头舔一舔地上的草丛。小玲开始感到一丝奇特的安慰：“它并不凶恶，也许我可以和它交朋友。”\n\n鼓起勇气，小玲一步一步走近怪兽。她轻声说道：“嗨，你好，你的名字是什么？”怪兽听到小玲的声音，转过头来，眼中流露出一种温暖的光芒。它似乎理解了小玲的话，慢慢地向她靠近，发出了一阵柔和低吟的声音，像是在回应她的问候。\n\n小玲心中涌起一阵欢喜，她渐渐忘记了刚才的恐惧。她慢慢伸出手，试探性地抚摸怪兽的鳞片，虽然有些粗糙，但却暖和如同阳光。怪兽用头轻轻蹭了蹭小玲的手，仿佛在邀请她一起玩耍。\n\n在那片森林里，小玲和怪兽像朋友一样快乐地嬉戏，草地上留下了她们的笑声和欢呼。时间在不知不觉中流逝，小玲深深体会到，每一个看似可怕的事物，背后都可能隐藏着温柔和友好。\n\n当夕阳西下，小玲知道是时候回家了。和怪兽依依惜别时，她心中不禁涌起一阵留恋：“我一定会再来的，你要等我哦！”怪兽轻轻点了点头，仿佛在回应着她。\n\n回家的路上，小玲的心情愉快无比，她暗下决心，明天还要再来这片森林，去寻找那个有趣的朋友。她明白了，勇气和开放的心灵，才能让她遇见世界上最美好的事物。'},
         'type': 'text'}],
       'created_at': 1727258232,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_iH1Ll2mLI0SdARhAEuec4tz4',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动'},
         'type': 'text'}],
       'created_at': 1727258147,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'}],
     'object': 'list',
     'first_id': 'msg_kIG2Cxn2lqJ42dGYmLUdNEen',
     'last_id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
     'has_more': False}



&emsp;&emsp;继续进行新一轮的提问。


```python
message_3 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇歌颂中国的文章"
)
```


```python
run_3 = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
```


```python
response_3 = client.beta.threads.messages.list(thread_id=thread.id)
response_3.to_dict()
```




    {'data': [{'id': 'msg_UX7AnTOJaU8r8S8gXq61qLlD',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '### 歌颂中国\n\n在五千年的历史长河中，中国这片广袤的土地，以其独特的文化、丰厚的底蕴和顽强的精神，谱写了一曲激动人心的华美乐章。她如一位睿智的长者，见证了历史的沉浮，承载了无数先贤的智慧与勇气。祖国，中国，你是我的骄傲！\n\n**雄伟壮丽的自然风光**\n\n走遍大好河山，从巍峨的长城到秀美的桂林山水，无不彰显着大自然的鬼斧神工。黄河奔腾，赋予了中华民族无尽的生命力；长江骤涌，连接着南北的和谐与繁荣。每一座山、每一条河流都在倾诉着历史的故事，交织出幅幅壮丽的画卷。我们的祖国以她独特的自然风光，展现出无与伦比的魅力，仿佛在告诉世人：“这里是中华儿女的家园！”\n\n**博大精深的文化底蕴**\n\n中国文化如一条涓涓细流，涵养着我们民族的灵魂。儒家思想的仁义礼智信，道教的自然哲学，佛教的慈悲智慧，构成了中华文化的丰厚底蕴。从古老的诗词歌赋到现代的文学艺术，每一笔每一划都流露出对生命的思考与对价值的追求。无论是李白的豪放，还是陶渊明的清逸，都是民族精神的真实写照。我们以文化为纽带，紧紧相连，代代相传。\n\n**不屈不挠的民族精神**\n\n中国人民历经磨难，却从未屈服。无论是历史上披荆斩棘的先民，还是抗击外侮的英雄，他们都以不屈的姿态，捍卫着祖国的尊严与自由。在新时代的浪潮中，一代代中华儿女以无畏的精神，奋发向上，追逐梦想。从“两弹一星”到载人航天，每一个成就的背后，都是无数奋斗者的辛勤付出与不懈努力。我们用汗水与智慧，书写着中国崛起的新篇章。\n\n**繁荣发展的现代中国**\n\n走进现代的中国，仿佛步入了一座繁华的城市。高楼大厦拔地而起，科技创新层出不穷，民众的生活水平不断提升。当我们在五光十色的城市夜景中漫步时，不禁感叹时代的巨变。中国的梦想与希望血脉相连，每一个努力拼搏的身影，都是祖国未来的栋梁。我们在新时代的浪潮中，心手相连，共同前行！\n\n**面向未来的中国**\n\n展望未来，中国将继续踏上更加辉煌的征途。我们要坚守民族精神，发扬团结奋斗的力量，推动科学技术的进步，实现可持续发展。面对全球化的挑战和机遇，我们的信念愈发坚定，那就是：不论未来多么不确定，只要团结一致，奋勇前行，必将迎来更加美好的明天。\n\n祖国，中国，这片热土承载着我们的梦想与希望。在你的怀抱中，我心生自豪。在新时代的号角声中，愿我中华儿女携手共进，书写壮丽的华章，创造更加辉煌的未来！让我们一同歌颂这一方土地，歌颂我们的祖国！在东方的晨光中，向着未来奔腾而去！'},
         'type': 'text'}],
       'created_at': 1727258571,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_gOulKAcAQ33LgA5tFkUs72Ms',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_sepBrjj6bou4jSW3pgUmhKY7',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '写一篇歌颂中国的文章'},
         'type': 'text'}],
       'created_at': 1727258568,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_kIG2Cxn2lqJ42dGYmLUdNEen',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在遥远的古代，天宫中的神仙们过着安详而宁静的生活。然而，一场动乱悄然降临，因为孙悟空，这位石猴，刚刚取得了无比强大的法力，决意要在天宫中大展宏图。一切就此拉开序幕，接下来他将以不可一世的姿态，翻天覆地。\n\n那一天，阳光明媚，天宫的众神正在举行盛大的庆典，庆祝天帝的长寿。突然，天宫外传来了震耳欲聋的吼声，随之而来的，是一个身影如电般飞驰而来，正是孙悟空。他化作一阵狂风，瞬间闯入了天宫的殿堂。\n\n“谁敢拦我！”他的声音如雷霆，震动着整个天宫。众仙一阵惊慌，纷纷朝四周奔逃。天兵天将们立刻反应过来，拿起武器，准备迎战。然而，面对这个曾经在花果山狂妄称王的石猴，他们内心的不安油然而生。\n\n孙悟空高高跃起，手中握着他的如意金箍棒，顺手挥动，金光闪烁，仿佛要撕裂天空。他的身形迅速变换，转瞬之间，变成了无数个猴影，让天兵天将无法捕捉到他的真实位置。 “看招！”他一声怒吼，金箍棒瞬间化作十倍的长度，狠狠地朝最近的天将砸去。\n\n“挡住他！”天将们纷纷聚拢，用力架住武器。然而，他们却只能听见“轰”的一声巨响，伴随着震耳欲聋的冲击波，天将们纷纷被击退，狼狈不堪。孙悟空趁机纵身而起，他的武技如风，任意游走于战场之上，天兵天将们根本无法抵挡。\n\n“碧空之上，谁敢与我一战？”孙悟空大喊，眼中燃烧着不屈的怒火。就在此时，天宫中的众神纷纷派出强大的儒雅仙神，试图拦住孙悟空的正义之路。那位名叫卷帘大将的神将，手持紫金钺，直抵云霄，气势攀升，蓦地冲向孙悟空。\n\n“你也想来试试？”孙悟空嘴角勾起一抹轻蔑的笑容，随即一跃而起，他的金箍棒再度变幻，直逼卷帘大将。两者在空中交锋，气劲如虹，斗技令人屏息以待。卷帘大将的钺与孙悟空的棒，在空中不断擦撞，四周的云彩被震得破碎，瞬间化为无数光点！\n\n战斗愈演愈烈，直到最终，卷帘大将终于被孙悟空的力道压制，他痛苦地跌落在地。紧接着，众天兵天将接踵而至，然而孙悟空已然势不可挡。他以无与伦比的速度和力量，扫荡整个天宫，众神惊吓退避，神仙也无暇应对。\n\n然而，天宫中的如来佛祖此时也注意到了这一切，他暗自思量，决定出手制止这场混乱。佛祖轻轻一抬手，蓦然间，整个天宫的气氛骤然一变。 shimmering 慢慢亮起，一道道金光闪烁，直逼孙悟空而来。孙悟空心中一凛，他意识到这位佛祖的威势非同小可，心中既惊又怒，“我倒要看看你有什么本事能够压住我！”\n\n孙悟空挺身而出，与如来佛祖展开了激烈的交手。他挥舞金箍棒，施展出百般法术，然而如来佛祖依然从容应对，仿佛这场战斗如行云流水，游刃有余。最终，孙悟空的力量被如来佛祖轻易化解，“你果然是强大的法力，但此处乃天宫，岂容你这般放肆！”\n\n一声喝止，孙悟空瞬间感受到一股无形的压力，犹如山岳压顶。他意识到自己在前力量上遇到了瓶颈，心中稍微不甘，但眼见眼前的如来佛祖，诸多感情蜂拥而至。“你为何要阻止我？”他问，眼中透露出不屈的挑战。\n\n“世间无恶，何苦与天斗？”如来佛祖温文尔雅，但其中的坚定和威严让孙悟空不得不动摇。经过一番斗智斗勇，最后孙悟空终于意识到，继续这样的战斗只能徒劳，反而会让自己陷入更深的绝境。\n\n“好吧，我认输，不过我一定会回来的！”孙悟空看着如来佛祖，消散了心中的怒火，松开了金箍棒，默默退回了自己的花果山。\n\n从此，他的心中留下一道无法磨灭的烙印，天宫虽好，但他明白自己的路还远未结束，未来还将有更多的冒险与挑战。大闹天宫，孙悟空不仅是一次蛮横的反抗，更是他追寻自我的一段传奇史诗。'},
         'type': 'text'}],
       'created_at': 1727258513,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_sSvOaN2szuGCHYS8AmTEtLHW',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_mcVj20d1FQDLX69oHQ2Ci9sH',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '写一篇关于孙悟空大闹天宫的精彩战斗故事'},
         'type': 'text'}],
       'created_at': 1727258509,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_RidBVSlednYxDJb83UpVehqH',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在一个阳光明媚的早晨，小女孩小玲决定走出家门，去附近的森林探险。她背上一个小背包，里面装着几块饼干和一瓶水，兴奋地踏上了通往森林的小道。一路上，鸟儿在树梢上欢快地鸣唱，花儿在微风中轻轻摇摆，整个世界都显得那么生机勃勃。\n\n走进森林后，阳光透过树叶洒下斑驳的光影，小玲感到了一丝神秘的气息。她不知道自己在森林里究竟会遇到什么，心里既期待又有些紧张。小玲喜欢冒险，然而未知的事物终究会让她心头泛起一丝怯意。\n\n就在她走得深入的时候，突然听见了一阵低沉的咕噜声。那声音像是在远处的某个地方回响，似乎带着一种奇异的韵律。小玲停下脚步，四处张望，试图找出声源。她的心跳开始加速，既想靠近去看个究竟，又有些害怕。\n\n她鼓起勇气，缓步走向声音传来的方向。越走越近，咕噜声渐渐清晰，夹杂着偶尔的树枝折断声。小玲的好奇心战胜了心中的恐惧，终于走到了一个开阔的空地上。然而，她目睹的景象令她大吃一惊：在空地中央，矗立着一只巨大的怪兽。\n\n这只怪兽浑身长满了色彩斑斓的鳞片，犹如大海中的珊瑚，反射出五光十色的光芒。它的眼睛像两个闪烁的宝石，注视着小玲，嘴巴张开时露出的长牙如同白色的柱子，显得既可怕又神秘。小玲的心中充满了恐惧和疑惑：“这是个什么生物？它会不会攻击我？”\n\n可是，出乎意料的是，怪兽似乎并不打算伤害她。它只是在空地上悠闲地活动，偶尔低下头来，用长长的舌头舔一舔地上的草丛。小玲开始感到一丝奇特的安慰：“它并不凶恶，也许我可以和它交朋友。”\n\n鼓起勇气，小玲一步一步走近怪兽。她轻声说道：“嗨，你好，你的名字是什么？”怪兽听到小玲的声音，转过头来，眼中流露出一种温暖的光芒。它似乎理解了小玲的话，慢慢地向她靠近，发出了一阵柔和低吟的声音，像是在回应她的问候。\n\n小玲心中涌起一阵欢喜，她渐渐忘记了刚才的恐惧。她慢慢伸出手，试探性地抚摸怪兽的鳞片，虽然有些粗糙，但却暖和如同阳光。怪兽用头轻轻蹭了蹭小玲的手，仿佛在邀请她一起玩耍。\n\n在那片森林里，小玲和怪兽像朋友一样快乐地嬉戏，草地上留下了她们的笑声和欢呼。时间在不知不觉中流逝，小玲深深体会到，每一个看似可怕的事物，背后都可能隐藏着温柔和友好。\n\n当夕阳西下，小玲知道是时候回家了。和怪兽依依惜别时，她心中不禁涌起一阵留恋：“我一定会再来的，你要等我哦！”怪兽轻轻点了点头，仿佛在回应着她。\n\n回家的路上，小玲的心情愉快无比，她暗下决心，明天还要再来这片森林，去寻找那个有趣的朋友。她明白了，勇气和开放的心灵，才能让她遇见世界上最美好的事物。'},
         'type': 'text'}],
       'created_at': 1727258232,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_iH1Ll2mLI0SdARhAEuec4tz4',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动'},
         'type': 'text'}],
       'created_at': 1727258147,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'}],
     'object': 'list',
     'first_id': 'msg_UX7AnTOJaU8r8S8gXq61qLlD',
     'last_id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
     'has_more': False}



&emsp;&emsp;这里我们抛出一个`Assistant API`的异常机制，即**当执行完上一轮的运行状态，马上在当前线程中追加新一轮的对话请求**，就会发生如下错误：


```python
message_4 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="请介绍一下中国国庆节的由来。"
)

run_4 = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
```


```python
message_5 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="请介绍一下中国中秋节的由来"
)
```


    ---------------------------------------------------------------------------

    BadRequestError                           Traceback (most recent call last)

    Cell In[80], line 1
    ----> 1 message_5 = client.beta.threads.messages.create(
          2   thread_id=thread.id,
          3   role="user",
          4   content="请介绍一下中国中秋节的由来"
          5 )


    File ~\anaconda3\envs\agent\Lib\site-packages\openai\resources\beta\threads\messages.py:88, in Messages.create(self, thread_id, content, role, attachments, metadata, extra_headers, extra_query, extra_body, timeout)
         86     raise ValueError(f"Expected a non-empty value for `thread_id` but received {thread_id!r}")
         87 extra_headers = {"OpenAI-Beta": "assistants=v2", **(extra_headers or {})}
    ---> 88 return self._post(
         89     f"/threads/{thread_id}/messages",
         90     body=maybe_transform(
         91         {
         92             "content": content,
         93             "role": role,
         94             "attachments": attachments,
         95             "metadata": metadata,
         96         },
         97         message_create_params.MessageCreateParams,
         98     ),
         99     options=make_request_options(
        100         extra_headers=extra_headers, extra_query=extra_query, extra_body=extra_body, timeout=timeout
        101     ),
        102     cast_to=Message,
        103 )


    File ~\anaconda3\envs\agent\Lib\site-packages\openai\_base_client.py:1260, in SyncAPIClient.post(self, path, cast_to, body, options, files, stream, stream_cls)
       1246 def post(
       1247     self,
       1248     path: str,
       (...)
       1255     stream_cls: type[_StreamT] | None = None,
       1256 ) -> ResponseT | _StreamT:
       1257     opts = FinalRequestOptions.construct(
       1258         method="post", url=path, json_data=body, files=to_httpx_files(files), **options
       1259     )
    -> 1260     return cast(ResponseT, self.request(cast_to, opts, stream=stream, stream_cls=stream_cls))


    File ~\anaconda3\envs\agent\Lib\site-packages\openai\_base_client.py:937, in SyncAPIClient.request(self, cast_to, options, remaining_retries, stream, stream_cls)
        928 def request(
        929     self,
        930     cast_to: Type[ResponseT],
       (...)
        935     stream_cls: type[_StreamT] | None = None,
        936 ) -> ResponseT | _StreamT:
    --> 937     return self._request(
        938         cast_to=cast_to,
        939         options=options,
        940         stream=stream,
        941         stream_cls=stream_cls,
        942         remaining_retries=remaining_retries,
        943     )


    File ~\anaconda3\envs\agent\Lib\site-packages\openai\_base_client.py:1041, in SyncAPIClient._request(self, cast_to, options, remaining_retries, stream, stream_cls)
       1038         err.response.read()
       1040     log.debug("Re-raising status error")
    -> 1041     raise self._make_status_error_from_response(err.response) from None
       1043 return self._process_response(
       1044     cast_to=cast_to,
       1045     options=options,
       (...)
       1049     retries_taken=options.get_max_retries(self.max_retries) - retries,
       1050 )


    BadRequestError: Error code: 400 - {'error': {'message': "Can't add messages to thread_phxLhtF42ZaswePBms3vVbKf while a run run_Ognzli64rArWvniwD8JlHdIG is active.", 'type': 'invalid_request_error', 'param': None, 'code': None}}


&emsp;&emsp;如上报错所示：这种运行方式存在的问题是：每一轮新的问题返回的结果中，其返回结果并不包含最新一轮问题的回复，而出现  `Can't add messages to thread_jr2IooEGckZTkg2CmhMZwRxj while a run run_ks1uZXfnhPKM5mQw22hi54Qx is active` 这样的异常信息，恰好反映了 `Run`的运行状态转移过程，一个细节的点是大家回看`run_2.to_dict()`返回的响应体内容，会发现所有的`status`都是处于`queued`状态，其根本原因还是在于：当一个线程还在执行上一个任务的时候，再次调用一个新的运行状态，一定是有问题的。这也就说明：当执行运行时，线程会锁定。在状态传递到终止状态之前，我们是无法向线程添加消息或对其执行另一次运行。

&emsp;&emsp;稍微等待一会再进行尝试，则会发现一切又恢复正常调用了。


```python
message_5 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="请介绍一下中国中秋节的由来"
)
```


```python
run_5 = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id,
)
```


```python
response_5 = client.beta.threads.messages.list(thread_id=thread.id)
```


```python
response_5.to_dict()
```




    {'data': [{'id': 'msg_6y7860KFwCPfUKmwDBjjwICG',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '中秋节，又称月圆节、团圆节，是中国重要的传统节日之一，通常在农历八月十五日庆祝。中秋节源远流长，蕴含着丰富的文化意义和深厚的民俗传统。以下是中秋节的由来与其相关的文化背景。\n\n### 中秋节的历史起源\n\n1. **古代农耕文化的延续**：中秋节的起源可以追溯到古代农耕社会。古人以八月为收获的季节，人们在这一天庆祝丰收，感恩大自然的恩赐。古代的祭月活动，最初是为了祈求丰收和安宁，后来逐渐演变为拜月、赏月的习俗。\n\n2. **嫦娥奔月的神话**：中秋节背后有着丰富的神话传说，其中最著名的是“嫦娥奔月”的故事。传说中，嫦娥是后羿的妻子，为了保护不死药，最终选择吞下药丸，飞往月宫。在中秋的夜晚，嫦娥在月亮上寂寞地思念着她的丈夫，形成了团圆与思念的主题，象征着人们对亲人的渴望和对美好生活的追求。\n\n3. **唐宋时期的普及**：中秋节作为一个正式的节日，始于唐代，并在宋代期间愈发盛行，逐渐成为了普遍的民俗活动。特别是在宋代，有了“月饼”的出现，成为节日庆祝的重要食品。同时，赏月、吃月饼、寄托情感的活动日益普及，形成了中秋节的基本习俗。\n\n### 中秋节的习俗\n\n中秋节的庆祝方式丰富多彩，各地的民俗活动各具特色，但共同的主题是在这个团圆的日子里，人与人之间的情感交流：\n\n1. **赏月**：中秋之夜，月亮最为圆满明亮，人们通常在晚上与家人团聚，一起赏月，分享对亲人和美好生活的祝愿。\n\n2. **吃月饼**：月饼是中秋节的传统食品，象征着团圆与和谐。不同地区的月饼口味各异，有的甜，有的咸，各具特色。\n\n3. **提灯笼**：在一些地方，特别是儿童会提着灯笼游玩，增加了节日的欢庆气氛。灯笼因为其象征意义，寄托着对未来的美好期望。\n\n4. **诗歌与文化**：中秋节也是文化的象征，古往今来，许多文人墨客借此机会吟咏月色，抒发思乡之情。如苏轼的《水调歌头·明月几时有》便是经典之作，表达了对月亮的眷恋与人间隔阂的感慨。\n\n### 现代中秋节\n\n如今，中秋节作为国家法定节假日之一，已成为家庭团聚、亲友相聚的重要时刻。许多城市会举办相关的文艺活动和庆祝活动，推广传统文化，增强民族认同感。无论是家人团聚、还是朋友相聚，中秋节都体现了“团圆”、“和谐”的重要意义。\n\n中秋节不仅仅是一个传统节日，更是中国文化的重要组成部分，承载着千年历史与人们的情感寄托，在节日的氛围中，人们共同感受着中华文化的博大精深。'},
         'type': 'text'}],
       'created_at': 1727258674,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_JNUExWKYaPqWshcavHhXuv9u',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_AGUByv2ZnSyG0FQBVI4MkPgb',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '请介绍一下中国中秋节的由来'},
         'type': 'text'}],
       'created_at': 1727258665,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_G9GzTuU3jChRiuikJcwW4AFc',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '中国国庆节，官方名称为“中华人民共和国成立纪念日”，每年的10月1日是这一重要节日。国庆节的由来可以追溯到1949年，当时中国社会正经历着深刻的变革与动荡。\n\n### 国庆节的历史背景\n\n1. **新中国成立**：1949年10月1日，经过长达数十年的抗战与内战，中国共产党在毛泽东的领导下，打败国民党，建立了中华人民共和国。这一事件标志着中国进入了一个新的历史时期，从此，国家开始走向独立、富强和现代化。\n\n2. **天安门广场的宣告**：在新中国成立的当天，毛泽东在北京天安门广场上庄严宣布：“中华人民共和国中央人民政府今天成立了！”这一历史性的讲话不仅宣布了新政府的成立，也表明了民族自信心的恢复与国家主权的重获。\n\n### 国庆节的设立\n\n在新中国成立的初期，为了回应全国人民对这个历史性时刻的庆祝愿望，政府决定将10月1日定为国庆节，并开始在这一天举行各种形式的庆祝活动。\n\n### 国庆节的庆祝活动\n\n国庆节成为了全国人民欢庆独立与团结的日子。每年这一时期，各地都会举办丰富多彩的庆祝活动，包括：\n\n- **阅兵仪式**：特别是在重要的周年纪念日，天安门广场经常举行盛大的阅兵仪式，展示国家的军事力量。\n- **文艺演出**：各类文艺演出、歌唱、舞蹈和文化活动展现了中华民族的独特魅力。\n- **焰火表演**：在国庆节晚上，许多城市都会举行焰火晚会，绚丽的烟花点亮夜空，象征着希望和团结。\n- **群众游行**：全国各地的市民会参与各种游行活动，表达对祖国的热爱和祝福。\n\n### 现代国庆节\n\n如今，国庆节已不仅是一个单纯的庆祝日，更是一种体现中华民族团结、和谐与自豪的象征。它也成为现代中国社会的重要文化符号，吸引了来自海内外的视线。国庆节期间的假期安排，使得人们有机会与家人团聚，出游放松，感受祖国的大好河山。\n\n通过国庆节的庆祝活动，人民不仅回顾新中国66年来的发展历程，也展望未来的发展目标，凝聚起全民族的力量，共同为实现中华民族伟大复兴的中国梦而努力。'},
         'type': 'text'}],
       'created_at': 1727258631,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_Ognzli64rArWvniwD8JlHdIG',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_0PjJNBLRcJ4yfKMah5WgUKdQ',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '请介绍一下中国国庆节的由来。'},
         'type': 'text'}],
       'created_at': 1727258629,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_UX7AnTOJaU8r8S8gXq61qLlD',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '### 歌颂中国\n\n在五千年的历史长河中，中国这片广袤的土地，以其独特的文化、丰厚的底蕴和顽强的精神，谱写了一曲激动人心的华美乐章。她如一位睿智的长者，见证了历史的沉浮，承载了无数先贤的智慧与勇气。祖国，中国，你是我的骄傲！\n\n**雄伟壮丽的自然风光**\n\n走遍大好河山，从巍峨的长城到秀美的桂林山水，无不彰显着大自然的鬼斧神工。黄河奔腾，赋予了中华民族无尽的生命力；长江骤涌，连接着南北的和谐与繁荣。每一座山、每一条河流都在倾诉着历史的故事，交织出幅幅壮丽的画卷。我们的祖国以她独特的自然风光，展现出无与伦比的魅力，仿佛在告诉世人：“这里是中华儿女的家园！”\n\n**博大精深的文化底蕴**\n\n中国文化如一条涓涓细流，涵养着我们民族的灵魂。儒家思想的仁义礼智信，道教的自然哲学，佛教的慈悲智慧，构成了中华文化的丰厚底蕴。从古老的诗词歌赋到现代的文学艺术，每一笔每一划都流露出对生命的思考与对价值的追求。无论是李白的豪放，还是陶渊明的清逸，都是民族精神的真实写照。我们以文化为纽带，紧紧相连，代代相传。\n\n**不屈不挠的民族精神**\n\n中国人民历经磨难，却从未屈服。无论是历史上披荆斩棘的先民，还是抗击外侮的英雄，他们都以不屈的姿态，捍卫着祖国的尊严与自由。在新时代的浪潮中，一代代中华儿女以无畏的精神，奋发向上，追逐梦想。从“两弹一星”到载人航天，每一个成就的背后，都是无数奋斗者的辛勤付出与不懈努力。我们用汗水与智慧，书写着中国崛起的新篇章。\n\n**繁荣发展的现代中国**\n\n走进现代的中国，仿佛步入了一座繁华的城市。高楼大厦拔地而起，科技创新层出不穷，民众的生活水平不断提升。当我们在五光十色的城市夜景中漫步时，不禁感叹时代的巨变。中国的梦想与希望血脉相连，每一个努力拼搏的身影，都是祖国未来的栋梁。我们在新时代的浪潮中，心手相连，共同前行！\n\n**面向未来的中国**\n\n展望未来，中国将继续踏上更加辉煌的征途。我们要坚守民族精神，发扬团结奋斗的力量，推动科学技术的进步，实现可持续发展。面对全球化的挑战和机遇，我们的信念愈发坚定，那就是：不论未来多么不确定，只要团结一致，奋勇前行，必将迎来更加美好的明天。\n\n祖国，中国，这片热土承载着我们的梦想与希望。在你的怀抱中，我心生自豪。在新时代的号角声中，愿我中华儿女携手共进，书写壮丽的华章，创造更加辉煌的未来！让我们一同歌颂这一方土地，歌颂我们的祖国！在东方的晨光中，向着未来奔腾而去！'},
         'type': 'text'}],
       'created_at': 1727258571,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_gOulKAcAQ33LgA5tFkUs72Ms',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_sepBrjj6bou4jSW3pgUmhKY7',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '写一篇歌颂中国的文章'},
         'type': 'text'}],
       'created_at': 1727258568,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_kIG2Cxn2lqJ42dGYmLUdNEen',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在遥远的古代，天宫中的神仙们过着安详而宁静的生活。然而，一场动乱悄然降临，因为孙悟空，这位石猴，刚刚取得了无比强大的法力，决意要在天宫中大展宏图。一切就此拉开序幕，接下来他将以不可一世的姿态，翻天覆地。\n\n那一天，阳光明媚，天宫的众神正在举行盛大的庆典，庆祝天帝的长寿。突然，天宫外传来了震耳欲聋的吼声，随之而来的，是一个身影如电般飞驰而来，正是孙悟空。他化作一阵狂风，瞬间闯入了天宫的殿堂。\n\n“谁敢拦我！”他的声音如雷霆，震动着整个天宫。众仙一阵惊慌，纷纷朝四周奔逃。天兵天将们立刻反应过来，拿起武器，准备迎战。然而，面对这个曾经在花果山狂妄称王的石猴，他们内心的不安油然而生。\n\n孙悟空高高跃起，手中握着他的如意金箍棒，顺手挥动，金光闪烁，仿佛要撕裂天空。他的身形迅速变换，转瞬之间，变成了无数个猴影，让天兵天将无法捕捉到他的真实位置。 “看招！”他一声怒吼，金箍棒瞬间化作十倍的长度，狠狠地朝最近的天将砸去。\n\n“挡住他！”天将们纷纷聚拢，用力架住武器。然而，他们却只能听见“轰”的一声巨响，伴随着震耳欲聋的冲击波，天将们纷纷被击退，狼狈不堪。孙悟空趁机纵身而起，他的武技如风，任意游走于战场之上，天兵天将们根本无法抵挡。\n\n“碧空之上，谁敢与我一战？”孙悟空大喊，眼中燃烧着不屈的怒火。就在此时，天宫中的众神纷纷派出强大的儒雅仙神，试图拦住孙悟空的正义之路。那位名叫卷帘大将的神将，手持紫金钺，直抵云霄，气势攀升，蓦地冲向孙悟空。\n\n“你也想来试试？”孙悟空嘴角勾起一抹轻蔑的笑容，随即一跃而起，他的金箍棒再度变幻，直逼卷帘大将。两者在空中交锋，气劲如虹，斗技令人屏息以待。卷帘大将的钺与孙悟空的棒，在空中不断擦撞，四周的云彩被震得破碎，瞬间化为无数光点！\n\n战斗愈演愈烈，直到最终，卷帘大将终于被孙悟空的力道压制，他痛苦地跌落在地。紧接着，众天兵天将接踵而至，然而孙悟空已然势不可挡。他以无与伦比的速度和力量，扫荡整个天宫，众神惊吓退避，神仙也无暇应对。\n\n然而，天宫中的如来佛祖此时也注意到了这一切，他暗自思量，决定出手制止这场混乱。佛祖轻轻一抬手，蓦然间，整个天宫的气氛骤然一变。 shimmering 慢慢亮起，一道道金光闪烁，直逼孙悟空而来。孙悟空心中一凛，他意识到这位佛祖的威势非同小可，心中既惊又怒，“我倒要看看你有什么本事能够压住我！”\n\n孙悟空挺身而出，与如来佛祖展开了激烈的交手。他挥舞金箍棒，施展出百般法术，然而如来佛祖依然从容应对，仿佛这场战斗如行云流水，游刃有余。最终，孙悟空的力量被如来佛祖轻易化解，“你果然是强大的法力，但此处乃天宫，岂容你这般放肆！”\n\n一声喝止，孙悟空瞬间感受到一股无形的压力，犹如山岳压顶。他意识到自己在前力量上遇到了瓶颈，心中稍微不甘，但眼见眼前的如来佛祖，诸多感情蜂拥而至。“你为何要阻止我？”他问，眼中透露出不屈的挑战。\n\n“世间无恶，何苦与天斗？”如来佛祖温文尔雅，但其中的坚定和威严让孙悟空不得不动摇。经过一番斗智斗勇，最后孙悟空终于意识到，继续这样的战斗只能徒劳，反而会让自己陷入更深的绝境。\n\n“好吧，我认输，不过我一定会回来的！”孙悟空看着如来佛祖，消散了心中的怒火，松开了金箍棒，默默退回了自己的花果山。\n\n从此，他的心中留下一道无法磨灭的烙印，天宫虽好，但他明白自己的路还远未结束，未来还将有更多的冒险与挑战。大闹天宫，孙悟空不仅是一次蛮横的反抗，更是他追寻自我的一段传奇史诗。'},
         'type': 'text'}],
       'created_at': 1727258513,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_sSvOaN2szuGCHYS8AmTEtLHW',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_mcVj20d1FQDLX69oHQ2Ci9sH',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [], 'value': '写一篇关于孙悟空大闹天宫的精彩战斗故事'},
         'type': 'text'}],
       'created_at': 1727258509,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_RidBVSlednYxDJb83UpVehqH',
       'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '在一个阳光明媚的早晨，小女孩小玲决定走出家门，去附近的森林探险。她背上一个小背包，里面装着几块饼干和一瓶水，兴奋地踏上了通往森林的小道。一路上，鸟儿在树梢上欢快地鸣唱，花儿在微风中轻轻摇摆，整个世界都显得那么生机勃勃。\n\n走进森林后，阳光透过树叶洒下斑驳的光影，小玲感到了一丝神秘的气息。她不知道自己在森林里究竟会遇到什么，心里既期待又有些紧张。小玲喜欢冒险，然而未知的事物终究会让她心头泛起一丝怯意。\n\n就在她走得深入的时候，突然听见了一阵低沉的咕噜声。那声音像是在远处的某个地方回响，似乎带着一种奇异的韵律。小玲停下脚步，四处张望，试图找出声源。她的心跳开始加速，既想靠近去看个究竟，又有些害怕。\n\n她鼓起勇气，缓步走向声音传来的方向。越走越近，咕噜声渐渐清晰，夹杂着偶尔的树枝折断声。小玲的好奇心战胜了心中的恐惧，终于走到了一个开阔的空地上。然而，她目睹的景象令她大吃一惊：在空地中央，矗立着一只巨大的怪兽。\n\n这只怪兽浑身长满了色彩斑斓的鳞片，犹如大海中的珊瑚，反射出五光十色的光芒。它的眼睛像两个闪烁的宝石，注视着小玲，嘴巴张开时露出的长牙如同白色的柱子，显得既可怕又神秘。小玲的心中充满了恐惧和疑惑：“这是个什么生物？它会不会攻击我？”\n\n可是，出乎意料的是，怪兽似乎并不打算伤害她。它只是在空地上悠闲地活动，偶尔低下头来，用长长的舌头舔一舔地上的草丛。小玲开始感到一丝奇特的安慰：“它并不凶恶，也许我可以和它交朋友。”\n\n鼓起勇气，小玲一步一步走近怪兽。她轻声说道：“嗨，你好，你的名字是什么？”怪兽听到小玲的声音，转过头来，眼中流露出一种温暖的光芒。它似乎理解了小玲的话，慢慢地向她靠近，发出了一阵柔和低吟的声音，像是在回应她的问候。\n\n小玲心中涌起一阵欢喜，她渐渐忘记了刚才的恐惧。她慢慢伸出手，试探性地抚摸怪兽的鳞片，虽然有些粗糙，但却暖和如同阳光。怪兽用头轻轻蹭了蹭小玲的手，仿佛在邀请她一起玩耍。\n\n在那片森林里，小玲和怪兽像朋友一样快乐地嬉戏，草地上留下了她们的笑声和欢呼。时间在不知不觉中流逝，小玲深深体会到，每一个看似可怕的事物，背后都可能隐藏着温柔和友好。\n\n当夕阳西下，小玲知道是时候回家了。和怪兽依依惜别时，她心中不禁涌起一阵留恋：“我一定会再来的，你要等我哦！”怪兽轻轻点了点头，仿佛在回应着她。\n\n回家的路上，小玲的心情愉快无比，她暗下决心，明天还要再来这片森林，去寻找那个有趣的朋友。她明白了，勇气和开放的心灵，才能让她遇见世界上最美好的事物。'},
         'type': 'text'}],
       'created_at': 1727258232,
       'metadata': {},
       'object': 'thread.message',
       'role': 'assistant',
       'run_id': 'run_iH1Ll2mLI0SdARhAEuec4tz4',
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'},
      {'id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
       'assistant_id': None,
       'attachments': [],
       'content': [{'text': {'annotations': [],
          'value': '写一篇关于一个小女孩在森林里遇到一只怪兽的故事。详细介绍她的所见所闻，并描述她的心里活动'},
         'type': 'text'}],
       'created_at': 1727258147,
       'metadata': {},
       'object': 'thread.message',
       'role': 'user',
       'run_id': None,
       'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf'}],
     'object': 'list',
     'first_id': 'msg_6y7860KFwCPfUKmwDBjjwICG',
     'last_id': 'msg_6lm5pzwVIfPemH6uBZAshsGZ',
     'has_more': False}



&emsp;&emsp;同理，在获取当前最新一轮对话的`assistant`回复时，如果当前的`Run`运行状态未进入`complete`状态，也会返回异常信息。


```python
response_5.data[0].content[0].text.value
```




    '中秋节是中华民族的重要传统节日之一，通常在农历八月十五庆祝。这一天，家家户户团聚在一起，共享月饼，赏月，象征着团圆和丰收。中秋节的由来源远流长，涵盖了丰富的历史与文化内涵。\n\n### 中秋节的由来\n\n1. **历史背景**：\n   中秋节的起源可追溯到古代的月亮崇拜活动。在古代中国，农民在丰收季节感谢月亮的恩泽，习惯在秋天的满月之夜进行祭月活动。早在周朝时期（公元前1046-256年），就已经有“中秋”的相关记载。\n\n2. **祭月的习俗**：\n   随着时间的推移，祭月活动逐渐演变为节日庆典，尤其在唐朝（618-907年），庆祝活动更加盛大。唐代诗人杜甫的诗句中就提到了中秋，这意味着中秋节的文化内涵已经深入人心。\n\n3. **团圆的象征**：\n   中秋节不仅是对丰收的庆祝，也是家人团聚的时刻。古人认为，圆月象征着团圆和完整，尤其在这一天，亲人们会聚在一起，共享丰盛的餐点，表达对彼此的思念与祝福。\n\n### 中秋节的传说\n\n中秋节的庆祝活动也伴随着许多美丽的传说。其中最著名的包括：\n\n- **嫦娥奔月**：\n  这个传说讲述的是月亮女神嫦娥不幸成为了孤独的月中仙子。她的丈夫后羿是一位英勇的射手，射下了九个太阳，为人间带来了光明。为了保护不死药，后羿的徒弟妒恨，迫使嫦娥饮下了药剂。她为了避免人间的纷扰，飞向了月球，成为了孤独的月亮女神。每逢中秋，嫦娥都会在这一天光顾人间，给人们带来祝福。\n\n- **吴刚伐桂**：\n  另一个传说是吴刚在月宫中砍伐月桂树。吴刚因犯错被贬到月宫，只能不停地砍伐月桂树，这棵树在被砍倒后会自动复生，吴刚永远无法完成他的任务。\n\n### 中秋节的习俗\n\n中秋节有许多传统习俗，包括：\n\n- **吃月饼**：\n  月饼是中秋节的代表性食物。不同地区有不同风味的月饼，通常以豆沙、莲蓉或鲜肉等为馅。人们在这一天互赠月饼，以表达祝福和团圆之情。\n\n- **赏月**：\n  中秋节的晚上，家人围坐在一起，仰望皎洁的明月，象征着团圆与和谐。同时，许多地方会举办赏月活动，举办诗词朗诵会或音乐会。\n\n- **灯笼**：\n  在一些地区，孩子们会提着灯笼，游玩于夜间，象征着逐渐长大的孩子和吉祥的来年。\n\n### 总结\n\n中秋节是中国传统文化的瑰宝，是团圆、思念与丰收的象征。它不仅是个人和家庭的节日，更承载了中华民族的共同情感。随着时代的发展，中秋节不断融入新的元素，但其核心价值——团圆、和谐与祝福，始终未变。这个节日让我们在繁忙的生活中不忘初心，珍惜与亲人朋友的每一刻。'



&emsp;&emsp;而如果稍等片刻在进行尝试，则会正常运行。


```python
response_5.data[0].content[0].text.value
```




    '中秋节是中华民族的重要传统节日之一，通常在农历八月十五庆祝。这一天，家家户户团聚在一起，共享月饼，赏月，象征着团圆和丰收。中秋节的由来源远流长，涵盖了丰富的历史与文化内涵。\n\n### 中秋节的由来\n\n1. **历史背景**：\n   中秋节的起源可追溯到古代的月亮崇拜活动。在古代中国，农民在丰收季节感谢月亮的恩泽，习惯在秋天的满月之夜进行祭月活动。早在周朝时期（公元前1046-256年），就已经有“中秋”的相关记载。\n\n2. **祭月的习俗**：\n   随着时间的推移，祭月活动逐渐演变为节日庆典，尤其在唐朝（618-907年），庆祝活动更加盛大。唐代诗人杜甫的诗句中就提到了中秋，这意味着中秋节的文化内涵已经深入人心。\n\n3. **团圆的象征**：\n   中秋节不仅是对丰收的庆祝，也是家人团聚的时刻。古人认为，圆月象征着团圆和完整，尤其在这一天，亲人们会聚在一起，共享丰盛的餐点，表达对彼此的思念与祝福。\n\n### 中秋节的传说\n\n中秋节的庆祝活动也伴随着许多美丽的传说。其中最著名的包括：\n\n- **嫦娥奔月**：\n  这个传说讲述的是月亮女神嫦娥不幸成为了孤独的月中仙子。她的丈夫后羿是一位英勇的射手，射下了九个太阳，为人间带来了光明。为了保护不死药，后羿的徒弟妒恨，迫使嫦娥饮下了药剂。她为了避免人间的纷扰，飞向了月球，成为了孤独的月亮女神。每逢中秋，嫦娥都会在这一天光顾人间，给人们带来祝福。\n\n- **吴刚伐桂**：\n  另一个传说是吴刚在月宫中砍伐月桂树。吴刚因犯错被贬到月宫，只能不停地砍伐月桂树，这棵树在被砍倒后会自动复生，吴刚永远无法完成他的任务。\n\n### 中秋节的习俗\n\n中秋节有许多传统习俗，包括：\n\n- **吃月饼**：\n  月饼是中秋节的代表性食物。不同地区有不同风味的月饼，通常以豆沙、莲蓉或鲜肉等为馅。人们在这一天互赠月饼，以表达祝福和团圆之情。\n\n- **赏月**：\n  中秋节的晚上，家人围坐在一起，仰望皎洁的明月，象征着团圆与和谐。同时，许多地方会举办赏月活动，举办诗词朗诵会或音乐会。\n\n- **灯笼**：\n  在一些地区，孩子们会提着灯笼，游玩于夜间，象征着逐渐长大的孩子和吉祥的来年。\n\n### 总结\n\n中秋节是中国传统文化的瑰宝，是团圆、思念与丰收的象征。它不仅是个人和家庭的节日，更承载了中华民族的共同情感。随着时代的发展，中秋节不断融入新的元素，但其核心价值——团圆、和谐与祝福，始终未变。这个节日让我们在繁忙的生活中不忘初心，珍惜与亲人朋友的每一刻。'



&emsp;&emsp;当然，对于单轮何时响应完成，在开发逻辑中我们肯定不能一直静默等待或者设置固定的等待时间（例如 3秒），正确的做法是做轮询更新。正如之前所说：`run_2.to_dict()`等不同轮次对话返回的响应体内容，所有的status都是处于queued状态，真实的开发逻辑应该是：**定时检索 `Run`对象，每次检索对象时检查运行状态，如果监测到完成，直接提取其结果。**

&emsp;&emsp;检索`Run`运行状态的过程，`Assistant API` 提供了一个 新的方法 `.beta.threads.runs.retrieve`，所以我们就可以借助这个 API 来实现上面提到的功能需求。完整代码如下：

> Retrieve run：https://platform.openai.com/docs/api-reference/runs/getRun


```python
message_6 = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇国庆节普天同庆的庆祝文章"
)
```


```python
# step 1. 创建运行（run）
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

# 打印运行的初始状态
print(f"run: {run}")

# 监控运行状态，这个循环持续检查运行的状态直到它完成。每次循环会重新获取运行的最新状态并打印，直到状态变为 completed。
while run.status !="completed":
  run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
  )
  print(f"run_status:{run.status}")


messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

# 获取并打印最终消息
print(f"Final Answer:{messages.data[0].content[0].text.value}")
```

    run: Run(id='run_9N5VTDdJ0HvqfOcWmusNNlk4', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', cancelled_at=None, completed_at=None, created_at=1727258757, expires_at=1727259357, failed_at=None, incomplete_details=None, instructions='You are an expert at writing excellent literature', last_error=None, max_completion_tokens=None, max_prompt_tokens=None, metadata={}, model='gpt-4o-mini-2024-07-18', object='thread.run', parallel_tool_calls=True, required_action=None, response_format='auto', started_at=None, status='queued', thread_id='thread_phxLhtF42ZaswePBms3vVbKf', tool_choice='auto', tools=[], truncation_strategy=TruncationStrategy(type='auto', last_messages=None), usage=None, temperature=1.0, top_p=1.0, tool_resources={})
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:in_progress
    run_status:completed
    Final Answer:### 国庆节普天同庆
    
    金秋十月，阳光明媚，带着丰收的喜悦和希望，我们迎来了中华人民共和国成立的庆典——国庆节。在这个特殊的日子里，祖国的每一个角落都被欢声笑语和节日的气氛所笼罩，普天同庆，庆祝我们伟大的祖国走过了辉煌的历程。
    
    **欢庆的氛围**
    
    国庆节是每一个中华儿女心中无比重要的日子。早晨，城市与乡村都被五彩斑斓的国旗装点得如诗如画，家家户户挂上了红灯笼，悬起了彩带，象征着喜庆与团圆。无论大街小巷，处处洋溢着欢乐的气息，行人们脸上都挂着自豪的微笑，大家在共同的庆祝中感受到亲如一家。
    
    **盛大的庆典**
    
    在国家的心脏——北京，天安门广场举行了盛大的阅兵式和文艺演出。耀眼的阳光照耀着庄严的天安门，军队整齐的步伐和响亮的口号回荡在空中，展示着中国人民的团结奋进和保家卫国的决心。整个广场如潮水般涌动的群众，齐声高唱国歌，表达着对祖国母亲的敬爱和骄傲。
    
    在全国各地，各类庆祝活动层出不穷。从文艺表演到乡村集市，从烟花晚会到各类赛事，丰富多彩的节目使得人们能够放下平日的忙碌，尽情享受这一份欢乐。此外，许多地方还举办了义务献血、环保志愿活动，展现出中华民族的团结与互助精神。
    
    **团圆与欢聚**
    
    国庆节不仅是欢庆的时刻，也是家人团聚的日子。许多人借着这个长假，回到故乡，和亲人一起聚餐、共度温馨时光。在圆桌边，人们分享着美味的佳肴，欢声笑语不断，诉说着对生活的感悟与对未来的憧憬。乡音未改，情谊依旧，每一个家庭都成为了这个节日的最温暖的舞台。
    
    **展望未来**
    
    国庆节是团结的象征，也是奋进的号召。站在历史的新起点，中华民族正以昂扬的姿态，走向更加美好的明天。我们深知，今日之成就来之不易，未来更需勤奋与拼搏。在这个国庆节的庆祝中，大家不仅仅是在庆贺过去的辉煌，更是在憧憬未来的宏伟蓝图。
    
    在全球化的浪潮中，我们的国家肩负着更多的机遇与挑战。让我们继续发扬艰苦奋斗、团结拼搏的精神，努力实现中华民族的伟大复兴，共同筑就更加光辉灿烂的明天。
    
    **总结**
    
    在这个国庆节，让我们共同祝愿我们的祖国繁荣昌盛、国泰民安。无论身在何处，心中都有一个不灭的信念——爱国心永远与我们同在，团结是我们不可或缺的力量。愿我们的祖国在未来的征途上更加璀璨辉煌，愿每一个中华儿女在这片土地上共同书写更加美好的篇章！国庆快乐！


&emsp;&emsp;如上显示的动态监测过程，大模型在实际生成响应时，其状态会持续保持`in progress`，只有完成全部响应后，才会将状态变更为`complete`，而我们在整个过程中只需要监测这个状态，就能够实时的获取到当前流程处于哪一个阶段。


```python
run.to_dict()
```




    {'id': 'run_9N5VTDdJ0HvqfOcWmusNNlk4',
     'assistant_id': 'asst_DQ41JJBwmGwhx8LOp6u76lST',
     'cancelled_at': None,
     'completed_at': 1727258769,
     'created_at': 1727258757,
     'expires_at': None,
     'failed_at': None,
     'incomplete_details': None,
     'instructions': 'You are an expert at writing excellent literature',
     'last_error': None,
     'max_completion_tokens': None,
     'max_prompt_tokens': None,
     'metadata': {},
     'model': 'gpt-4o-mini-2024-07-18',
     'object': 'thread.run',
     'parallel_tool_calls': True,
     'required_action': None,
     'response_format': 'auto',
     'started_at': 1727258757,
     'status': 'completed',
     'thread_id': 'thread_phxLhtF42ZaswePBms3vVbKf',
     'tool_choice': 'auto',
     'tools': [],
     'truncation_strategy': {'type': 'auto', 'last_messages': None},
     'usage': {'completion_tokens': 820,
      'prompt_tokens': 4701,
      'total_tokens': 5521},
     'temperature': 1.0,
     'top_p': 1.0,
     'tool_resources': {}}



&emsp;&emsp;同样，在最终的返回结果中，`status`状态也不再是`queued`，而是`complete`。 需要说明的一点时：这种实现方式也是我们在实际构建项目时会采用的流程，请大家务必掌握。

&emsp;&emsp;至此，我们已经借助`Assistant API`提供的功能方法，完整地实现了顺序通信流程图所示的全部过程。即 👇

<div align=center><img src="https://muyu001.oss-cn-beijing.aliyuncs.com/img/image-20240923165414686.png" width=80%></div>

&emsp;&emsp;当然，`Assistant API` 的能力远不止于此。其更加复杂和自定义的高级功能，如内置工具、Function Calling的使用、以及流式输出等，都是面向产品级应用开发的核心技术。不过，这些进阶的技术应用都是基于我们今天介绍的知识的一种扩展。我们将在接下来的课程中，详细介绍每个部分的重点。但在此之前，我们还需要继续探讨`Assistant API` 的其他重要接口使用方法和应用技巧。

# 3. Assistant API的关键接口

&emsp;&emsp;在这一小节中，我们来进一步补充`Assistant API`的其他关键接口。

## 3.1 Assistant 对象相关接口

&emsp;&emsp;首先第一个是通过`.assistants.list` 方法，可以查看到所有已创建的`Assistatant API`对象。创建`Assistant`实例`OpenAI`目前没有明确的数量限制。其参数如下：

> List assistants：https://platform.openai.com/docs/api-reference/assistants/listAssistants

| 参数名    | 类型     | 可选性   | 默认值  | 描述                                                                                              |
|-----------|----------|----------|---------|---------------------------------------------------------------------------------------------------|
| limit     | integer  | 可选     | 20      | 返回对象的数量限制，范围为 1 到 100，默认值为 20。                                              |
| order     | string   | 可选     | desc    | 根据对象的创建时间戳进行排序，asc 为升序，desc 为降序。                                         |
| after     | string   | 可选     | -       | 用于分页的游标，after 是一个对象 ID，用于定义在列表中的位置。例如，若请求列表并接收到 100 个对象，以 obj_foo 结尾，后续调用可以包含 after=obj_foo，以获取下一页列表。 |
| before    | string   | 可选     | -       | 用于分页的游标，before 是一个对象 ID，用于定义在列表中的位置。例如，若请求列表并接收到 100 个对象，以 obj_foo 结尾，后续调用可以包含 before=obj_foo，以获取上一页表。 |



```python
my_assistants = client.beta.assistants.list(
    order="desc",
    limit="20",
)

my_assistants
```




    SyncCursorPage[Assistant](data=[Assistant(id='asst_DQ41JJBwmGwhx8LOp6u76lST', created_at=1727257878, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_8vKN3493CeU7hXeIQehmuGvN', created_at=1727246130, description=None, instructions="You're a weather data analyst. When asked for data information, write and run code to answer the question", metadata={}, model='gpt-4o-mini-2024-07-18', name='Data Engineer', object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0), Assistant(id='asst_Dr2MOXMYsrMJfw1yXQVdlI02', created_at=1727245668, description=None, instructions="You're a weather data analyst. When asked for data information, write and run code to answer the question", metadata={}, model='gpt-4o', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0), Assistant(id='asst_Tn6kggssCtE3Sm1OOt2wR8u1', created_at=1727245438, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini-2024-07-18', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0), Assistant(id='asst_ImmqBsjcXjx7QFSvrNnXp8bE', created_at=1727244786, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini-2024-07-18', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0), Assistant(id='asst_YVVrTWIleUp1kW1LfSivLMGx', created_at=1727242799, description=None, instructions='You are a professional large model technician and apply your basic knowledge to answer large model related questions', metadata={}, model='gpt-4o-mini-2024-07-18', name='Large language model technical assistant', object='assistant', tools=[FileSearchTool(type='file_search', file_search=FileSearch(max_num_results=None, ranking_options=FileSearchRankingOptions(ranker='default_2024_08_21', score_threshold=0.0)))], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=ToolResourcesFileSearch(vector_store_ids=['vs_mvo18j1hnb0KdLIYHmfgp4GV'])), top_p=1.0), Assistant(id='asst_hwo9M8PZ1rb5paYc1HIxbrZ6', created_at=1727240192, description=None, instructions='You are a professional large model technician and apply your basic knowledge to answer large model related questions', metadata={}, model='gpt-4o-mini-2024-07-18', name='Large language model technical assistant', object='assistant', tools=[FileSearchTool(type='file_search', file_search=FileSearch(max_num_results=None, ranking_options=FileSearchRankingOptions(ranker='default_2024_08_21', score_threshold=0.0)))], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=ToolResourcesFileSearch(vector_store_ids=['vs_azZDjx3FJSYRuTSWnvMy3YK5'])), top_p=1.0), Assistant(id='asst_cMQnlR2yIgBk10BDVWPcr4O8', created_at=1727231557, description=None, instructions="You're a famous rapper", metadata={}, model='gpt-4o', name='Rapper', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_as4c0KkmhlQdgGbWG1RKkTtP', created_at=1727174681, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_qlUKlRhgjgxnQt0p8U3k6a77', created_at=1727158169, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_eLvnLNmFdWSIBsZrnmXd5Aj9', created_at=1727158110, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_NluLwyvHjnaWdV7fnS5o0TrZ', created_at=1727099705, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0), Assistant(id='asst_DSeBLisl39hJxnk7ATRgA0ZU', created_at=1727078114, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_ycpzIFtcdFDlawP9Kq44Mlxp', created_at=1727075672, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_8IKLyUXHtwtYceWvScPqIJQU', created_at=1727072023, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0), Assistant(id='asst_BzQOahZoTWv1ijn86a0NCYjy', created_at=1724223112, description=None, instructions=None, metadata={}, model='gpt-4o', name=None, object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0)], object='list', first_id='asst_DQ41JJBwmGwhx8LOp6u76lST', last_id='asst_BzQOahZoTWv1ijn86a0NCYjy', has_more=False)




```python
my_assistants.data
```




    [Assistant(id='asst_DQ41JJBwmGwhx8LOp6u76lST', created_at=1727257878, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_8vKN3493CeU7hXeIQehmuGvN', created_at=1727246130, description=None, instructions="You're a weather data analyst. When asked for data information, write and run code to answer the question", metadata={}, model='gpt-4o-mini-2024-07-18', name='Data Engineer', object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0),
     Assistant(id='asst_Dr2MOXMYsrMJfw1yXQVdlI02', created_at=1727245668, description=None, instructions="You're a weather data analyst. When asked for data information, write and run code to answer the question", metadata={}, model='gpt-4o', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0),
     Assistant(id='asst_Tn6kggssCtE3Sm1OOt2wR8u1', created_at=1727245438, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini-2024-07-18', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0),
     Assistant(id='asst_ImmqBsjcXjx7QFSvrNnXp8bE', created_at=1727244786, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini-2024-07-18', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0),
     Assistant(id='asst_YVVrTWIleUp1kW1LfSivLMGx', created_at=1727242799, description=None, instructions='You are a professional large model technician and apply your basic knowledge to answer large model related questions', metadata={}, model='gpt-4o-mini-2024-07-18', name='Large language model technical assistant', object='assistant', tools=[FileSearchTool(type='file_search', file_search=FileSearch(max_num_results=None, ranking_options=FileSearchRankingOptions(ranker='default_2024_08_21', score_threshold=0.0)))], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=ToolResourcesFileSearch(vector_store_ids=['vs_mvo18j1hnb0KdLIYHmfgp4GV'])), top_p=1.0),
     Assistant(id='asst_hwo9M8PZ1rb5paYc1HIxbrZ6', created_at=1727240192, description=None, instructions='You are a professional large model technician and apply your basic knowledge to answer large model related questions', metadata={}, model='gpt-4o-mini-2024-07-18', name='Large language model technical assistant', object='assistant', tools=[FileSearchTool(type='file_search', file_search=FileSearch(max_num_results=None, ranking_options=FileSearchRankingOptions(ranker='default_2024_08_21', score_threshold=0.0)))], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=ToolResourcesFileSearch(vector_store_ids=['vs_azZDjx3FJSYRuTSWnvMy3YK5'])), top_p=1.0),
     Assistant(id='asst_cMQnlR2yIgBk10BDVWPcr4O8', created_at=1727231557, description=None, instructions="You're a famous rapper", metadata={}, model='gpt-4o', name='Rapper', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_as4c0KkmhlQdgGbWG1RKkTtP', created_at=1727174681, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_qlUKlRhgjgxnQt0p8U3k6a77', created_at=1727158169, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_eLvnLNmFdWSIBsZrnmXd5Aj9', created_at=1727158110, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_NluLwyvHjnaWdV7fnS5o0TrZ', created_at=1727099705, description=None, instructions='You are a personal math tutor. When asked a math question, write and run code to answer the question.', metadata={}, model='gpt-4o-mini', name=None, object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0),
     Assistant(id='asst_DSeBLisl39hJxnk7ATRgA0ZU', created_at=1727078114, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_ycpzIFtcdFDlawP9Kq44Mlxp', created_at=1727075672, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_8IKLyUXHtwtYceWvScPqIJQU', created_at=1727072023, description=None, instructions='You are an expert at writing excellent literature', metadata={}, model='gpt-4o-mini-2024-07-18', name='Good writer', object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0),
     Assistant(id='asst_BzQOahZoTWv1ijn86a0NCYjy', created_at=1724223112, description=None, instructions=None, metadata={}, model='gpt-4o', name=None, object='assistant', tools=[], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=None, file_search=None), top_p=1.0)]




```python
for assistant_id in my_assistants.data:
    print(f"assistant_id:{assistant_id.id}")
```

    assistant_id:asst_DQ41JJBwmGwhx8LOp6u76lST
    assistant_id:asst_8vKN3493CeU7hXeIQehmuGvN
    assistant_id:asst_Dr2MOXMYsrMJfw1yXQVdlI02
    assistant_id:asst_Tn6kggssCtE3Sm1OOt2wR8u1
    assistant_id:asst_ImmqBsjcXjx7QFSvrNnXp8bE
    assistant_id:asst_YVVrTWIleUp1kW1LfSivLMGx
    assistant_id:asst_hwo9M8PZ1rb5paYc1HIxbrZ6
    assistant_id:asst_cMQnlR2yIgBk10BDVWPcr4O8
    assistant_id:asst_as4c0KkmhlQdgGbWG1RKkTtP
    assistant_id:asst_qlUKlRhgjgxnQt0p8U3k6a77
    assistant_id:asst_eLvnLNmFdWSIBsZrnmXd5Aj9
    assistant_id:asst_NluLwyvHjnaWdV7fnS5o0TrZ
    assistant_id:asst_DSeBLisl39hJxnk7ATRgA0ZU
    assistant_id:asst_ycpzIFtcdFDlawP9Kq44Mlxp
    assistant_id:asst_8IKLyUXHtwtYceWvScPqIJQU
    assistant_id:asst_BzQOahZoTWv1ijn86a0NCYjy


&emsp;&emsp;接下来比较关键，且在工程化应用中比较常用的是`.beta.assistants.retrieve()` 方法。之前提到过，一个 `Assistant` 对象实例可以视为一个用户或助手，在构建多用户/多代理的应用系统时，我们必须确保不同用户之间的信息相互隔离且独立。因此，这个方法在实际应用中会被频繁使用。`.beta.assistants.retrieve()`方法通过传入指定的`assistant id`，即可切换到不同的`Assistant`对象实例下执行后续的操作。代码如下所示：


```python
# 这个 id 替换为大家在自己实践过程中显示出来的 任一 assis id 即可
my_assistant = client.beta.assistants.retrieve("asst_8vKN3493CeU7hXeIQehmuGvN")
print(my_assistant)
```

    Assistant(id='asst_8vKN3493CeU7hXeIQehmuGvN', created_at=1727246130, description=None, instructions="You're a weather data analyst. When asked for data information, write and run code to answer the question", metadata={}, model='gpt-4o-mini-2024-07-18', name='Data Engineer', object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0)


&emsp;&emsp;除此之外，对于一个已经创建好的`Assistant`对象，我们还可以更改其初始化设定，比如使用的模型、身份背景和任务目标，这可以通过`.beta.assistants.update()`方法来实现。比如：


```python
my_updated_assistant = client.beta.assistants.update(
  "asst_8vKN3493CeU7hXeIQehmuGvN",
  instructions="You're a famous rapper",
  name="Rapper",
  model="gpt-4o"
)

print(my_updated_assistant)
```

    Assistant(id='asst_8vKN3493CeU7hXeIQehmuGvN', created_at=1727246130, description=None, instructions="You're a famous rapper", metadata={}, model='gpt-4o', name='Rapper', object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0)


&emsp;&emsp;在`.beta.assistants.update()`方法中可用的参数与`.beta.assistants.create()` 方法保持一致，意味着初始化`Assistant`对象的所有内容均可以被进行更新。


```python
my_assistant = client.beta.assistants.retrieve("asst_8vKN3493CeU7hXeIQehmuGvN")
print(my_assistant)
```

    Assistant(id='asst_8vKN3493CeU7hXeIQehmuGvN', created_at=1727246130, description=None, instructions="You're a famous rapper", metadata={}, model='gpt-4o', name='Rapper', object='assistant', tools=[CodeInterpreterTool(type='code_interpreter')], response_format='auto', temperature=1.0, tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None), top_p=1.0)


&emsp;&emsp;**这个方法在工程化开发中是比较常用的，通过更新`Assistant`的基础信息，就可以根据实时的任务需求灵活调整对话/代理的身份设定，使其在不同场景下响应出更好的效果。**

&emsp;&emsp;最后，就是`Assistant`对象的删除操作，这可以通过`.beta.assistants.delete()`方法，传入指定的`assistant id`进行删除。代码如下：


```python
response = client.beta.assistants.delete("asst_21B2vdnzwmWMW70UGldP4eox")
print(response)
```

    AssistantDeleted(id='asst_21B2vdnzwmWMW70UGldP4eox', deleted=True, object='assistant.deleted')


&emsp;&emsp;当返回`deleted=True`，则说明已经删除成功。

## 3.2 Thread 对象相关接口

&emsp;&emsp;`Thread`对象的相关操作和`Assistant`基本一样，`Assistant API`也提供了选择指定线程、更新线程信息和删除指定线程的三个操作。首先，选择指定线程，是通过`.beta.threads.retrieve()`方法，传入指定的`thread id`即可。但是，我们并不能像`.beta.assistants.list()` 一样通过某个方法来获取到所有的`thread id`列表，`OpenAI` 并没有提供。**而在工程化开发阶段，我们往往是把`thread id` 存储在本地的关系型数据库中，同时建立 `assistant id`、`thread id` 与 `message id` 三者的关联关系。这一部分我们将在接下来的课程中详细介绍。

&emsp;&emsp;这里为了演示接口方法，我们直接查找一个已知的`thread id`进行测试。首先，指定线程的话，代码如下所示：


```python
thread.id
```




    'thread_phxLhtF42ZaswePBms3vVbKf'




```python
my_thread = client.beta.threads.retrieve(thread_id=thread.id)
print(my_thread)
```

    Thread(id='thread_phxLhtF42ZaswePBms3vVbKf', created_at=1727258023, metadata={}, object='thread', tool_resources=ToolResources(code_interpreter=ToolResourcesCodeInterpreter(file_ids=[]), file_search=None))


&emsp;&emsp;删除指定线程，则需要调用`.beta.threads.delete()`方法，同时也只需要传入指定的`thread id`，代码如下：


```python
response = client.beta.threads.delete(thread_id=thread.id)
print(response)
```

    ThreadDeleted(id='thread_phxLhtF42ZaswePBms3vVbKf', deleted=True, object='thread.deleted')


&emsp;&emsp;当返回`deleted=True`，则说明已经删除成功。

&emsp;&emsp;而至于`Thread`的信息更新，涉及到`Assistant metadata`（元数据）的使用，我们将在下一小节的课程中结合实际的场景展开讨论。

## 3.3 Messages 对象相关接口

&emsp;&emsp;`Assistant API` 提供了一种使用`list`方法列出线程中消息的便捷方法。但这里有一点需要说明的是：在工程化开发中，在处理包含数百条消息的线程时，需要用到分页，否则可能导致严重的网络 I/O 问题。

&emsp;&emsp;这里我们新创建一个线程并执行两轮对话，构建测试数据。


```python
thread = client.beta.threads.create()
```


```python
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇国庆节普天同庆的庆祝文章"
)
```


```python
assistant.id
```




    'asst_DQ41JJBwmGwhx8LOp6u76lST'




```python
run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)
```


```python
message = client.beta.threads.messages.create(
  thread_id=thread.id,
  role="user",
  content="写一篇孙悟空大闹天宫的文章"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)
```


```python
thread.id
```




    'thread_dstOmhwqyx1g1RzX3ftM7Tim'



&emsp;&emsp;通过`.beta.threads.messages.list()`方法并传入指定线程id，查看其所有的会话记录。


```python
thread_messages = client.beta.threads.messages.list(thread_id=thread.id)
print(thread_messages.data)
```

    [Message(id='msg_22YNQLsNI5kH3ignCRvrqQTj', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='孙悟空大闹天宫，这是中国古典名著《西游记》中最为人熟知的一个传奇故事。这个故事不仅展示了孙悟空的机智和勇敢，也深刻揭示了反抗权威、追求自由的主题。\n\n故事的开端，孙悟空出生于花果山，他从石头中破裂而出，悟得了无上道理，成为斗战胜佛。他在师父菩提祖师的教导下，学习了高强的武艺和各种法术，拥有了如意金箍棒和七十二变的本领。猴子们生活得快乐而自由，然而，这种平静并没有持续太久。\n\n随后，孙悟空因性格倔强，不甘于平凡，决定向天庭寻求更高的地位。他首先被封为“齐天大圣”，这个称号使他意气风发。然而，天庭的封号只是名义上的荣耀，孙悟空很快察觉到，他在天庭并没有真正受到重视。这令他十分愤怒，于是，他决定大闹天宫，以此来寻求对自己地位的认可。\n\n孙悟空大闹天宫的过程简直让人瞠目结舌。面对天庭的众神，他毫不畏惧，展现了他无比强大的战斗力。从捣毁天宫的桃园到击败天兵天将，孙悟空的每一次出手都令众神惊恐万分。尤其是他那根如意金箍棒，光是摆动一下便可掀起狂风巨浪，空气中那么一丝冷凝的气氛也为之改变。\n\n面对孙悟空的强大，天庭不得不想尽各种方法来捉拿他。玉帝请来了如来佛祖，以他的智慧和慈悲来对付孙悟空。最后，在如来佛祖的引导下，孙悟空被困在五指山下，虽然一时失败，但他的叛逆精神却让后来的故事中注入了更多的勇气和激情。\n\n孙悟空大闹天宫的故事传递了反叛与追求自由的深刻内涵。孙悟空不甘被压迫，他用自己的方式回应强权，为那些被忽视的生灵发声。这个形象深受人们喜爱，它代表了对传统观念的挑战，对自由和尊严的渴望。\n\n最终，虽然孙悟空在五指山下被禁锢，但他的精神并未消失。此后，他在唐僧的指引下踏上取经之路，经历种种磨难，逐渐成长，最终实现了自我价值。这不仅是个人的成长，也是对整个社会的一种反思与启示。\n\n孙悟空大闹天宫的故事，如今依然在耳边回响。他的形象不仅存在于故事中，也成为无数人心中的自由象征，他的叛逆精神激励着一代又一代人去追求更美好的未来。在这个喧嚣的现代社会中，让我们继续传承、发扬这种奋发有为的精神，勇敢地面对挑战，寻求属于自己的那片天空。'), type='text')], created_at=1727259237, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_C2o4FYFeHJOpStdmawjHI6yy', status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_NPk910I9gqo3McV2LyiFysVS', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='写一篇孙悟空大闹天宫的文章'), type='text')], created_at=1727259236, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_8fQ9eatSlRB6WZjCOcBLbYJO', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='国庆节是中华民族一年一度的盛大节日，象征着伟大祖国的诞生和繁荣。每当这一天，祖国的大江南北都洋溢着喜悦的氛围，五彩的烟花、吉祥的舞狮、热闹的游行，无不展现着国人对祖国的热爱与祝福。\n\n在这特别的日子里，城市和乡村被国旗装点得格外鲜艳。街头巷尾、楼宇平台、公园广场，到处可见那鲜艳的红旗和金色的国徽，仿佛在向每一位市民诉说着国家的辉煌历程。越来越多人走上街头，分享这份喜悦，亲朋好友聚在一起，交流着过去一年的点滴，共同展望未来的美好。孩子们欢快地嬉戏游戏，老人们则亲切地聊着家长里短，洋溢着温暖的社区气息。\n\n在这一天，许多地方会举办丰富多彩的庆祝活动。各大城市的广场上，汇聚了来自四面八方的人们，他们共同欣赏激动人心的文艺演出，有如歌舞、民族乐器演奏等，展现出中华文化的博大精深。有的地方还会组织国庆游行，队伍中既有身着各式民族服装的表演者，也有士兵扛枪列队，展示出国家的军威与和平。人们用热烈的掌声回应着、用欢快的笑声传递着对祖国的热爱。\n\n在这个特殊的日子里，网络上也充满了祝福的声音。人们通过社交媒体分享国庆的快乐，表达对祖国繁荣富强的美好愿景。志愿者们走上街头，为抗击疫情和支持社会发展贡献自己的力量，展现着新时代青年人的责任与担当。\n\n国庆节不仅是庆祝的时刻，也是让我们思考的时刻。我们感恩那些为国家的发展和繁荣奉献的人们，无论是历史长河中的英雄先烈，还是现代社会中无数默默奉献的劳动者。在这个日子里，我们应更加铭记历史，珍惜当下，努力为实现中华民族的伟大复兴贡献自己的力量。\n\n国庆佳节，普天同庆。愿我们的祖国在未来的岁月里，繁荣昌盛，人民幸福，让每一个中国人都有机会追梦圆梦。让我们共同祝愿，祖国永远年轻，未来更加辉煌！'), type='text')], created_at=1727259218, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_uour0tewEbwsUy580jZlzmlW', status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_lnuJ88Eme33fawsA5GzCqt8o', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='写一篇国庆节普天同庆的庆祝文章'), type='text')], created_at=1727259210, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim')]


&emsp;&emsp;分页的关键参数是after 、 before和limit 。
- after ：用于分页的光标。它指定定义在列表中位置的对象 ID。例如，如果发出列表请求并收到 100 个以obj_foo结尾的对象，则后续调用可以包含after=obj_foo来获取下一页。
- before ：与after类似，但它指定获取上一页的对象 ID。
- limit ：每页返回的对象数量。限制范围为 1 到 100，默认值为 20。

&emsp;&emsp;使用`after`参数进行分页检索线程中所有消息的代码如下：

- 初始化：初始化 OpenAI 客户端并设置要从中获取消息的线程的thread_id 。
- 分页循环：使用while循环不断获取消息页面，直到没有更多消息返回。
- 获取消息：调用 list 方法与 limit 和 after 参数。最初， after 是 None 。
- 扩展列表：将获取的消息添加到all_messages列表中。
- 更新游标：将after游标更新为当前批次中最后一条消息的 ID。
- Break Condition ：当没有更多消息返回时退出循环。


```python
thread_id = thread.id
all_messages = []

# 每个请求允许的最大限制
limit = 100  
after = None

while True:
    response = client.beta.threads.messages.list(thread_id, limit=limit, after=after)
    messages = response.data
    if not messages:
        break
    all_messages.extend(messages)

    # 将'after'游标设置为最后一条消息的ID
    after = messages[-1].id  

print(f"Total messages retrieved: {len(all_messages)}")
```

    Total messages retrieved: 4



```python
print(f"Total messages retrieved: {all_messages}")
```

    Total messages retrieved: [Message(id='msg_22YNQLsNI5kH3ignCRvrqQTj', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='孙悟空大闹天宫，这是中国古典名著《西游记》中最为人熟知的一个传奇故事。这个故事不仅展示了孙悟空的机智和勇敢，也深刻揭示了反抗权威、追求自由的主题。\n\n故事的开端，孙悟空出生于花果山，他从石头中破裂而出，悟得了无上道理，成为斗战胜佛。他在师父菩提祖师的教导下，学习了高强的武艺和各种法术，拥有了如意金箍棒和七十二变的本领。猴子们生活得快乐而自由，然而，这种平静并没有持续太久。\n\n随后，孙悟空因性格倔强，不甘于平凡，决定向天庭寻求更高的地位。他首先被封为“齐天大圣”，这个称号使他意气风发。然而，天庭的封号只是名义上的荣耀，孙悟空很快察觉到，他在天庭并没有真正受到重视。这令他十分愤怒，于是，他决定大闹天宫，以此来寻求对自己地位的认可。\n\n孙悟空大闹天宫的过程简直让人瞠目结舌。面对天庭的众神，他毫不畏惧，展现了他无比强大的战斗力。从捣毁天宫的桃园到击败天兵天将，孙悟空的每一次出手都令众神惊恐万分。尤其是他那根如意金箍棒，光是摆动一下便可掀起狂风巨浪，空气中那么一丝冷凝的气氛也为之改变。\n\n面对孙悟空的强大，天庭不得不想尽各种方法来捉拿他。玉帝请来了如来佛祖，以他的智慧和慈悲来对付孙悟空。最后，在如来佛祖的引导下，孙悟空被困在五指山下，虽然一时失败，但他的叛逆精神却让后来的故事中注入了更多的勇气和激情。\n\n孙悟空大闹天宫的故事传递了反叛与追求自由的深刻内涵。孙悟空不甘被压迫，他用自己的方式回应强权，为那些被忽视的生灵发声。这个形象深受人们喜爱，它代表了对传统观念的挑战，对自由和尊严的渴望。\n\n最终，虽然孙悟空在五指山下被禁锢，但他的精神并未消失。此后，他在唐僧的指引下踏上取经之路，经历种种磨难，逐渐成长，最终实现了自我价值。这不仅是个人的成长，也是对整个社会的一种反思与启示。\n\n孙悟空大闹天宫的故事，如今依然在耳边回响。他的形象不仅存在于故事中，也成为无数人心中的自由象征，他的叛逆精神激励着一代又一代人去追求更美好的未来。在这个喧嚣的现代社会中，让我们继续传承、发扬这种奋发有为的精神，勇敢地面对挑战，寻求属于自己的那片天空。'), type='text')], created_at=1727259237, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_C2o4FYFeHJOpStdmawjHI6yy', status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_NPk910I9gqo3McV2LyiFysVS', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='写一篇孙悟空大闹天宫的文章'), type='text')], created_at=1727259236, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_8fQ9eatSlRB6WZjCOcBLbYJO', assistant_id='asst_DQ41JJBwmGwhx8LOp6u76lST', attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='国庆节是中华民族一年一度的盛大节日，象征着伟大祖国的诞生和繁荣。每当这一天，祖国的大江南北都洋溢着喜悦的氛围，五彩的烟花、吉祥的舞狮、热闹的游行，无不展现着国人对祖国的热爱与祝福。\n\n在这特别的日子里，城市和乡村被国旗装点得格外鲜艳。街头巷尾、楼宇平台、公园广场，到处可见那鲜艳的红旗和金色的国徽，仿佛在向每一位市民诉说着国家的辉煌历程。越来越多人走上街头，分享这份喜悦，亲朋好友聚在一起，交流着过去一年的点滴，共同展望未来的美好。孩子们欢快地嬉戏游戏，老人们则亲切地聊着家长里短，洋溢着温暖的社区气息。\n\n在这一天，许多地方会举办丰富多彩的庆祝活动。各大城市的广场上，汇聚了来自四面八方的人们，他们共同欣赏激动人心的文艺演出，有如歌舞、民族乐器演奏等，展现出中华文化的博大精深。有的地方还会组织国庆游行，队伍中既有身着各式民族服装的表演者，也有士兵扛枪列队，展示出国家的军威与和平。人们用热烈的掌声回应着、用欢快的笑声传递着对祖国的热爱。\n\n在这个特殊的日子里，网络上也充满了祝福的声音。人们通过社交媒体分享国庆的快乐，表达对祖国繁荣富强的美好愿景。志愿者们走上街头，为抗击疫情和支持社会发展贡献自己的力量，展现着新时代青年人的责任与担当。\n\n国庆节不仅是庆祝的时刻，也是让我们思考的时刻。我们感恩那些为国家的发展和繁荣奉献的人们，无论是历史长河中的英雄先烈，还是现代社会中无数默默奉献的劳动者。在这个日子里，我们应更加铭记历史，珍惜当下，努力为实现中华民族的伟大复兴贡献自己的力量。\n\n国庆佳节，普天同庆。愿我们的祖国在未来的岁月里，繁荣昌盛，人民幸福，让每一个中国人都有机会追梦圆梦。让我们共同祝愿，祖国永远年轻，未来更加辉煌！'), type='text')], created_at=1727259218, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='assistant', run_id='run_uour0tewEbwsUy580jZlzmlW', status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim'), Message(id='msg_lnuJ88Eme33fawsA5GzCqt8o', assistant_id=None, attachments=[], completed_at=None, content=[TextContentBlock(text=Text(annotations=[], value='写一篇国庆节普天同庆的庆祝文章'), type='text')], created_at=1727259210, incomplete_at=None, incomplete_details=None, metadata={}, object='thread.message', role='user', run_id=None, status=None, thread_id='thread_dstOmhwqyx1g1RzX3ftM7Tim')]


&emsp;&emsp;这种检索方式在对实时性要求较高的系统中是比较常用的有效减少响应时间的小技巧，建议大家进行尝试和使用。

&emsp;&emsp;以上就是今天课程的全部内容。而在掌握了基础的`Assistant API`构建方法后，下一小节我们将进一步学习和实践其更广泛用于开发应用系统的高阶用法。 
