# Kaldi笔记
## 训练mono hmm-gmm的过程

* gmm-init-mono 初始化单因素训练用的GMM参数和tree
* compile-train-graphs 为每个训练语音文件的抄本编译出hclg的graph.

* align-equal-compiled 对train graph做相等的对齐.
猜测:
如果没有多音字,每个训练的utt生成的hclg只有有的单个输入路径 
如果字典有多音字,每个训练的utt生成的hclg有多个输入路径 
语音序列长T帧, 输入路径上N个音素状态, 每个音素状态占T/N帧.
若有多条路径?

* gmm-acc-stats-ali
根据对齐信息计算gmm的统计量, 产生acc数据.

* show-alignments 展示对齐信息, 状态序列, 音素序列.

* gmm-est
根据acc统计量信息更新gmm的参数

* gmm-align-compiled 利用utt的fst和新的模型, 对train样本的状态序列做对齐.


训练过程
```

gmm-init-mono --out=tree
compile-train-graphs --in=tree --out=train.fst.gz

align-equal-compiled --in=train.fst.gz --out=train.ali
gmm-acc-stats-ali --in=train.ali --out==train.acc
gmm-est --in=train.acc --in=old.mdl --out==new.mdl

for i in 0-N:
    gmm-align-compiled --in=train.fst.gz --in=new.mdl --out=train.ali
    gmm-acc-stats-ali --in=train.ali --out==train.acc
    old.mdl = new.mdl
    gmm-est --in=train.acc --in=old.mdl --out==new.mdl
end
```




## Feature

AddDeltas分析
* AddDeltas  bin文件
* ComputeDeltas 对外接口
* DeltaFeatures::Process 真正的计算

SubVector获得一个Vector的局部引用.可以修改其引用的部分.

deltas特征是当前帧和前后帧的加权和.normalize by L2.

windows =1 下一帧减去上一帧
order=1 (-1,0,1)
order=2 (-1,-1,0,1,1) 

windows =2 
order=1 (-2,-1,0,1,2)
order=2 (2,-3,-5,-3,0,3,5,3,2) 

看一下mfcc,plp,fbank的代码设计,结构都一样的.

## IVector

## Lattice

用于判别式学习，解码图里除了当前路径，也要有其他路径。
当前路径是分子，所有路径是分母。寻找一个梯度方向，提升当前路径的概率，同时降低其他路径的概率

Train HMM-GMM时只生成transcript对应的fst，即只有正确文本的路径。
那Lattice怎么生成？

## HmmTopology
HmmTopology由一组HmmState构成
HmmState
* self_loop_pdf_class 
* forward_pdf_class 
* transition 是一个数组，每个元素是(next_state, prob)对.

注意pdf_class和pdf_id的区别，pdf_class是HmmTopology中在对phone HMM中的各arc的绑定用的。
pdf_id是经过决策树具类后的pdf索引.


## Transition-id

Kaldi里HCLG解码图的input label并不是决策树绑定后的pdf-id, 而是一个transition-id。
每个transition-id对应一个transition-model，其概念类似于一个senone(或者认为是其上一条transition arc)，
transition-model包括了 哪个phone， 哪个state， 哪个transition arc，以及对应的pdf-id.

transition-model = [transition-state, transition-index]
transition-state = [phone, state, self_loop_pdf_id, forward_pdf_id]

Kaldi中的HMM，emiting是在状态上的，但是其却更像是一种状态emiting和边上emiting的折中方案。
其区分了self_loop_pdf_id，forward_pdf_id，一个状态上的发生自跳转（只能有一个），则emit分布使用self_loop_pdf_id，
若发生跳出跳转（可以有多个），则均使用emit分布使用forward_pdf_id.

这种方式等价于边上emiting，但是所有跳出边share同一个emit分布。

神经网络的输出class是pdf-id，在解码时，需要计算所有transition-id的am得分，计算方法是，先把transition-id转换为其对应的pdf-id，然后获取对应的神经网络输出概率。

## Graph
HCLG的编译 http://vpanayotov.blogspot.com/2012/06/kaldi-decoding-graph-construction.html#c-fst

## Decoder

* DecodableInterface

## egs学习
* Librispeech
* Aishell
## 参考资料

* http://vpanayotov.blogspot.com/2012/06/kaldi-decoding-graph-construction.html#c-fst
* http://jrmeyer.github.io/
* https://github.com/oplatek/kaldi-thesis
* http://white.ucc.asn.au/Kaldi-Notes/


