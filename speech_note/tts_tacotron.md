# Tacotron系列论文笔记

- [x] TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS
- [x] Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
- [ ] List item Uncovering Latent Style Factors for Expressive Speech Synthesis
- [ ] Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron
- [ ] Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis
- [ ] Transfer Learning from Speaker Verification to Multispeaker Text-To-Speech Synthesis
- [ ] Predicting Expressive Speaking Style From Text in End-to-End Speech Synthesis
- [ ] Semi-Supervised Training for Improving Data Efficiency in End-to-End Speech Synthesis
- [ ] Hierarchical Generative Modeling for Controllable Speech Synthesis
- [ ] Disentangling Correlated Speaker and Noise for Speech Synthesis via Data Augmentation and Adversarial Factorization


目标：通过学习Tacotron系列论文，
了解语音合成领域的研究方向，研究方法，专用术语，评估方法和当前的系统性能

## 基本的E2E语音合成系统介绍

参数TTS的系统结构为 音频特征生成 + 声码器。
参考WAVENET: A GENERATIVE MODEL FOR RAW AUDIO的附录

其中，音频特征生成模块，输入是待合成的文本，输出为音频特征序列。
传统的声码器（vocoder)是基于语音信号处理方法，将频域特征还原回时域信号。2016年google提出的基于神经网络的声码器Wavenet，后来做了进一步优化，提升了合成音频质量并加快了计算速度，目前可以完全替代传统声码器。

Google TTS组的工作 [tacotron](https://google.github.io/tacotron/)

17年初到18年底，Google发表了一系列利用神经网络进行音频特征生成的方法，极大的提升了合成的质量，工作包括
* 一种基于Encoder-Decoder神经网络结构的参数TTS中的声学模型，称之为Tacotron
* 利用神经网络建模风格（style）信息，允许指定style
* 利用文本中获得style信息
* 多说话人

## 1.Tacotron开篇之作

*TACOTRON: TOWARDS END-TO-END SPEECH SYNTHESIS*

摘要

A text-to-speech synthesis system typically consists of multiple stages,
such as a text analysis frontend, an acoustic model and an audio synthesis module.
Build-ing these components often requires extensive domain expertise and may contain brittle design choices. 
In this paper, we present Tacotron, an end-to-end generative text-to-speech model that synthesizes speech directly from characters.
Given <text, audio> pairs, the model can be trained completely from scratch with random initialization.
We present several key techniques to make the sequence-to-sequence framework perform well for this challenging task.
Tacotron achieves a 3.82 subjective 5-scale mean opinion score on US English,
outperforming a production parametric system in terms of naturalness.
In addition, since Tacotron generates speech at the frame level, it’s substantially faster than sample-level autoregressive methods.

专业术语

parametric synthesis system 参数语音合成系统
* text analysis frontend
* an acoustic model
* an audio synthesis module

mean opinion score - MOS TTS里的主观评测指标，有1-5分选项，专业测听人员进行打分

frame-level 帧级别，对于8K采样的声音，如果帧移是10ms，每秒800帧。每帧根据帧长对应多个采样点，比如帧长25ms，8k的数据，每帧对应采样点个数为200个。
从帧级别的特征，可以生成采样点级别的信号。
samlple-level 采样点级别，对于8K采样的声音，每秒生成8000个采样点。


Tacotron是Encoder-Decoder 结构的网络。

Encoder = WordEmbeding+PreNet+CBHG

CBHG

Convolution Bank + Highway + bidirectional-GRU

Convolution Bank 是不同width的1-D filters，每个宽度的bank包含一组filters，每一时间点上所有filter的输出拼接起来。
比如，如果输入序列维度是128\*T，使用16个不同宽度的bank（宽度分别为1-16），若每种宽度的bank都有256个filter，stride=1，做好padding，则每个bank输出是256*T序列。
将所有bank拼接起来，最终输出是（256\*16）\* T的序列。

对于宽度为k的filter，则一个filter的参数有k\*128个.假设每宽度k有256个filter（256个输出channel）
总共的参数量为sum_k\*128\*256,若k取1-16，即一个CBHG层的参数量为17\*8\*128\*256=4400000。
CBHG输出每一时刻的维度为256\*16

CBHG中还使用了res连接，通过Convolution Bank的序列，最后使用一个输出channel和CBHG的输入序列channel大小一样的卷积，使得卷积后维度和原始输入相同，从而可以相加。

整个CBHG操作可以看作是序列的一种特征变化。原始的输入序列上每一个时刻点的特征仅是自身的信息，而通过CBank可以捕获不同窗长度内上下文时文本的信息，RNN也可以获得长距离上下文的信息，
从而经过变换后的特征序列，每个时刻上的特征出了包含当前时刻本身信息，还包含了当前时刻在此上下文情况下的信息。



Decoder

Decoder部分分两个网络
* 第一部分是PreNet+Attention RNN网络，该网络输出mel谱参数。
* 第二部分是一个CBHG网络，mel谱参数再经过该CBHG网络得到线性谱参数。
* 优化时同时使用这个两个输出和对应标注的误差作为优化目标。

线性谱是原语音帧的完整频谱表示。
梅尔谱特征是一个变换到梅尔域并且加三角窗且降低分辨率的更低维表示。

为了加速训练，Decoder中的PreNet+Attention RNN网络每个时刻输出连续多帧（比如3帧）的特征。

假设每帧是40个输出，则使用一个128*120的全联接层， 其中128为GRU的输出维度，120 = 40*3. 训练时

对于输出多帧的谱参数，只把每个时间点输出的多帧中的最后一帧作为下一个时间的输入。
```
In inference, at decoder step t, the last frame of the r predictions is fed as input to the decoder at step t + 1.
Note that feeding the last prediction is an ad-hoc choice here，we could use all r predictions. 
During training, we always feed every r-th ground truth frame to the decoder. 
```

Decoder的第一时刻点的输入为一个全零向量。

PreNet+Attention RNN直接输出线性谱,效果不好，论文中有图片比较通过PreNet+Attention RNN直接输出线性谱和
PreNet+Attention RNN输出mel谱然后再经过CBHG网络输出线性谱，后者看起来共振峰清晰很多。


Attention Align的图
decoder中的输出和encoder中的输出做attention，因为TTS的输出和输入是同样的顺序（语音识别也是同序的，机器翻译是不同序的），
因此，因此我们期望attention align的权重图，对于每个输出都主要align在某个输入时刻点上，且是平滑变化的，反映到图上时应该是一个
清晰的从左到右上升的台阶形状。  

Tacotron的MOS值好于当时最好的参数模型。使用了Griffin-Lim的Vocoder，效果比当时最好的拼接系统略差。
Tacotron 3.82 ± 0.085  
Parametric 3.69 ± 0.109
Concatenative 4.09 ± 0.119

## 2.Tacotron+Wavenet
Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions

This paper describes Tacotron 2, a neural network architecture for speech synthesis directly from text. 
The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character embeddings to mel-scale spectrograms, 
followed by a mod-ified WaveNet model acting as a vocoder to synthesize time-domain waveforms from those spectrograms. 
Our model achieves a mean opinion score (MOS) of 4.53 comparable to a MOS of 4.58 for professionally recorded speech. 
To validate our design choices, we present ablation studies of key components of our system and
evaluate the im-pact of using mel spectrograms as the conditioning input to WaveNet instead of linguistic, duration, and F0 features.
We further show that using this compact acoustic intermediate representation allows for a significant reduction in the size of the WaveNet architecture.

此论文也被称为Tacotron2，其对Tacotron开篇之作的中的系统做了如下改进：
* 简化了CBHG，去掉了其中的Highway网络
* 声学模型网络输出mel特征（而不是线性谱特征+F0）
* vocoder从Griffin-Lim换做了Wavenet，
* MOS直接到达4.53，逼近真人发音的4.58MOS值


## 3.带风格Style的合成方法  


Uncovering Latent Style Factors for Expressive Speech Synthesis

Towards End-to-End Prosody Transfer for Expressive Speech Synthesis with Tacotron

Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End Speech Synthesis

Predicting Expressive Speaking Style From Text in End-to-End Speech Synthesis