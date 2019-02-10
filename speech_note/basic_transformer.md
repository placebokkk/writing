# Attention/Transformer/Transformer XL

## Transformer

Attention
* Attention可以抽象为一个Q（query）和一组[(K，V)]（key，value). 
* 给定一个Q，根据Q和K的相关程度（attention度量方法），选择K对应的V。
* 和数据库检索不同，并不是只使用相关度最大的K对应的V，而是每个V都使用。通过度量公式计算Q和K的相关程度，和Q越相关的K其对应的V的权重越大，将所有的V加权求和。
* Q，K，V都是向量。且很多结构里使用的Attention机制，V就是K本身。

Multihead Attention
* Multihead是指，同时使用多组attention进行计算。
* 因为输入都是Q，[(K，V)].如果直接用Q，K，V来计算，每组attention机制算出的结果是一样的。
* 每组attention将K，V，Q分别做线性变化（线性变换的参数是网络的参数一部分），相当于计算信息在某个线性子空间上的attention，从而每组得到的attention结果是不同的。各组计算的结果再拼接起来作为最终结果。

Inter-Attention
* 在attention最初应用于NN时，Q是decoder中的hidden输出，K，V是encoder中的hidden输出（且K和V一样）.因此Q和K实际是两个不同序列上的信息，我称之为Inter-Attention，或者External-Attention。

Self-Attention
* 是指attention不跨越decoder和encoder，在encoder序列内或者decoder序列内进行。也可以叫Intra-Attention，Internal-Attental。
* 比如encoder中序列的每一时刻的某层信息（比如文本的embedding）都会和其他各时刻做attention操作，从而每个时刻都可以直接捕获其他时刻的信息。
* 在Self-attention中，Q和K是属于同一个序列的。

Positional Encoding（PE）
* self-attention不像RNN或者CNN可以捕捉到序列的顺序性，因此当顺序是重要的时候，可以通过Positional Encoding（PE）来为数据加入位置相关的信息。可以使用正余弦函数来构建，每一维选择不同的角频率，给定位置则PE值是确定的，且不同位置的PE值一定不一样。PE和输入直接相加。

Masked Multihead Self-Attention
* Decoder-Block里有个Masked Multihead Self-Attention中的Masked是指做attention时只和当前时刻之前的时刻做attention。因为decoder时输出（也是下一时刻的输入）从左往右依次产生的，在t时刻时，后面的数据还未产生。


Transformer的整体结构
* Transformer = Encoder + Decoder
* Encoder=Positional encoding + [Encoder-Block] * N 
* Encoder-Block=[Res+Multihead Self-Attention] + [Res+Dense]
* Decoder=Positional encoding + [Decoder-Block] * N
* Decoder-Block=[Res+Masked Multihead Self-Attention] + [Res+Multihead Inter-Attention] + [Res+Dense]

## Transformer的应用
### 各种端到端的序列转换任务
* 自然语言处理
    * 开山论文即展示在机器翻译和Constituency Parsing上应用达到state-of-the-art。
* 语音识别：
    * self-attention比较适合自然语言处理任务中希望捕获长距离上下文影响的情况。语音识别输入是帧级别的声学特征，在encoder阶段self-attention意义不大。decode阶段，目前的E2E识别系统建模单元一般都没到词级别（中文是音节/音素，字母类语言用音素/字符/音节/子词），也没必要使用覆盖特别长的上下文的self-attention。
    * 不过也有学者尝试用Transformer来做识别，Speech-transformer:A no-recurrence sequence-to-sequence model for speech recognition
    * pytorch代码实现 https://github.com/kaituoxu/Speech-Transformer
    * 大部分语音识别场景要求streaming，直接用encoder-decoder架构不适合。
* 语音合成的输入是文本，可以考虑应用self-attention捕获长距离上下文。
    * 微软的文章 https://arxiv.org/pdf/1809.08895.pdf
    * 语音合成不像nlp的任务需要关注特别长的上下文，tacotron中的conv bank本身覆盖的上下文长度（从1到16）已经足够长了。感觉使用self-attention还是conv bank都差不多。

### BERT
* OpenAI的Generative Pre-trained Transformer（GPT）最早Transformer来进行LM训练，该模型从左往右预测下一个词，类似于Transformer的Decoder部分，只使用了词的单侧上下文。
* BERT希望使用self-attention且能同时利用左右的上下文，但是直接用双侧的self-attention会使得在预测下一个词的时，输入信息中直接能看到下一个词是什么，模型无法得到有效学习。为了解决这个问题引入了masked LM的训练方法。对于要预测的词，将输入序列中对应的词置为一个特殊字符'[masked]', 这样输入就看不到待预测的词的信息了。
* BERT还增加了一个预测一个句子是不是另一个句子的下一个句子的任务来训练模型（是不是这个标注可以直接从语料免费获取，不需要额外标注）。将两个句子拼起来，用'[sep]'特殊字符分割，然后句首增加'[cls]'特殊字符，用'[cls]'的输出经过一个前向分类网络作为预测结果。
* 网络参数配置:
    * BERT-Base :12-layer+ 12-heads + 768-hidden，110M parameters
    * BERT-Large : 24-layer, 1024-hidden, 16-heads, 340M parameters
* 语言模型天生不需要人工标注，有海量训练数据。训练出的模型可作为其他任务的特征提取模块。将原文本序列转换成一个等长新序列。在几乎所有任务上都得到了很大的提升。
* 和word2vec直接为每个词提供一个表示（查表，不需要计算时间）不同，BERT是将每个词根据其所在上下完将其转为一个向量表示（称为Contextualized word-embeddings），优点是更好的利用了词当前所处上下文，缺点是需要在线计算且计算量不小。

## Transformer的优化
* Transformer XL
* Evoled Transformer

## 资料
### Attention
explain
* Attention/NTM/Transformer/SNAIL等 https://lilianweng.github.io/lil-log/2018/06/24/attention-attention.html

survey
* https://arxiv.org/pdf/1811.05544.pdf

### Transformer
paper
* https://arxiv.org/pdf/1706.03762.pdf

explain
* 傻瓜版图文讲解 http://jalammar.github.io/illustrated-transformer/
* 论文总结1 https://hub.packtpub.com/paper-in-two-minutes-attention-is-all-you-need/
* 论文总结2 http://mlexplained.com/2017/12/29/attention-is-all-you-need-explained/
* 论文讲解版带pytorch代码 http://nlp.seas.harvard.edu/2018/04/03/attention.html#model-architecture
* Google官方报告 https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html

video and slides
* https://www.youtube.com/watch?v=rBCqOTEfxvg
* https://drive.google.com/file/d/0B8BcJC1Y8XqobGNBYVpteDdFOWc/view

### BERT
* Google官方代码 https://github.com/google-research/bert
* 图文并茂，还顺带解释了一些列预训练表示模型 http://jalammar.github.io/illustrated-bert/
* 原始论文 https://arxiv.org/pdf/1810.04805.pdf

### Transformer XL
paper
* https://arxiv.org/pdf/1901.02860.pdf

code
* https://github.com/kimiyoung/transformer-xl

