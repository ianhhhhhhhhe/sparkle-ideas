# CNN 模型结构
![LeNet-5.jpeg](https://cdn.nlark.com/yuque/0/2023/jpeg/741660/1679643552139-7d626fd0-2893-4787-aae4-6e908752a782.jpeg#averageHue=%23d8d8d7&clientId=u1cef90bd-ce3f-4&from=drop&id=u5ca8254e&name=LeNet-5.jpeg&originHeight=239&originWidth=800&originalType=binary&ratio=2&rotation=0&showTitle=false&size=37125&status=done&style=none&taskId=u1cc6fe77-6fc4-44eb-9e4f-5c03da6f833&title=)
# CNN 模型的不足
在阅读文章时，上下文的关联对于人们理解文章的意义和情感有着非常重要的帮助，每次阅读文章中的语句时不会单单的对单条语句进行理解，还会关联前文中隐含的情感关联。传统的CNN模型无法对上下文的情感关联等进行记忆和处理，因此我们需要一种方式，将处理过的数据中隐含的信息传递保存在模型中。

---

# RNN 模型
RNN[5][12]是一类人工神经网络，其中节点之间的连接可以创建一个循环，允许一些节点的输出影响对相同节点的后续输入。
其具体表现形式如下：
$h_t = \tanh(x_t W_{ih}^T + b_{ih} + h_{t-1}W_{hh}^T + b_{hh})$
![RNN.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679279235055-4802c7b5-dce0-4174-87cb-8e24a93e2918.png#averageHue=%23000000&clientId=ue20f455d-4537-4&from=drop&id=u0cb2f195&name=RNN.png&originHeight=267&originWidth=800&originalType=binary&ratio=2&rotation=0&showTitle=false&size=22211&status=done&style=none&taskId=u4f0cd55d-494e-46a9-9eb6-72765788631&title=)
例如在预测“the clouds are in the _**sky**”_这句话的时候，语句中前几个词的信息会包含在hidden state中传递给最后一层，因此不需要额外信息就可以推断出最后一个词应该为_**sky**。_在短词句中，RNN的记忆还是可靠的，然而在一些长短落的处理中可能就不那么可靠了。
在《信号与系统》中，将一个系统的输出仅仅决定于该时刻的输入的系统成为无记忆系统；而记忆系统，则是具有保存或存贮不是当前时刻输入信息的功能。这对后续解释 Transformer 模型的记忆系统具有非常重要的意义。

---

# RNN 长期记忆的问题
例如有这么一段话“I grew up in France… I speak fluent _French_.”其中要预测最后一个单词大概是要补充的是一种语言，但是因为中间可能包含了很长的一段信息，要预测需要回溯到比较久的文本中去，两端相关联的文本距离较远的时候RNN的便无法学习与关联这些信息。
理论上，RNN绝对有能力处理这种“长期依赖性”。人类可以仔细地为它们选择参数来解决这种形式的问题。遗憾的是，在实践中，RNN似乎无法学习它们。Hochreiter（1991）[13]和Bengio等人（1994）[14]对这个问题进行了深入探讨，他们发现了这可能很困难的一些非常根本的原因。

---

# LSTM(Long short-term memory) 与 GRU(Gated recurrent unit)
## LSTM 模型
![LSTM.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679279248233-d934d6e6-f385-4bec-ac53-9668b8b64fab.png#averageHue=%2353504a&clientId=ue20f455d-4537-4&from=drop&id=ue888333e&name=LSTM.png&originHeight=876&originWidth=1280&originalType=binary&ratio=2&rotation=0&showTitle=false&size=69768&status=done&style=none&taskId=u5c0a962f-cf81-46db-be19-0934bd66cef&title=)
$\begin{array}{ll} \\
            i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\
            f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\
            g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\
            o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\
            c_t = f_t \odot c_{t-1} + i_t \odot g_t \\
            h_t = o_t \odot \tanh(c_t) \\
        \end{array}$
LSTM[6][11]模型通常会包含一个cell，一个forget gate，一个input gate，一个ouput gate。
forget gate 用于控制cell中的信息，哪些需要被遗忘，哪些可以传递给模型的下一层。input gate 决定新添加哪些信息到cell里。hidden state将和输入经过tanh函数后，与input gate相乘，得出我们要更新到state的更新内容。
输出是基于cell state与输入，先是一个sigmoid 层，再与经过tanh的cell相乘。
这里面cell相当于一个长期记忆，通过forget gate决定遗忘某些长期记忆的内容并用当前属性覆盖，hidden state的部分其实与普通RNN模型是一样的，其功能便是短期记忆。最终输出由长期记忆与短期记忆共同决定。
因此模型可以总结为三个阶段[15]:

1. 忘记阶段：通过forget gate选择性保留部分长期记忆
2. 选择记忆阶段：通过input gate进行选择性记忆
3. 输出阶段：根据当前状态进行输出
## GRU模型
![GRU.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679279257951-96e2f3ee-15c2-40bb-8963-6287ccb9e926.png#averageHue=%23ed974c&clientId=ue20f455d-4537-4&from=drop&id=uecd2dd05&name=GRU.png&originHeight=356&originWidth=800&originalType=binary&ratio=2&rotation=0&showTitle=false&size=41091&status=done&style=none&taskId=uec0fa992-31aa-4c73-a8c9-282291d8062&title=)
$\begin{array}{ll}
            r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr}) \\
            z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz}) \\
            n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn})) \\
            h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
        \end{array}$
GRU[10]模型中包含一个reset gate，一个update gate和一个new gate
相较于LSTM通过forget gate 和 input gate 进行遗忘和选择性记忆，GRU模型通过进行遗忘，通过进行记忆的方式更新长期记忆。
可以看到记忆和遗忘是包含联动机制，遗忘的部分会通过记忆进行补充，以保持一种“恒定”的状态[16].
# LSTM 和 GRU 的问题
虽然 LSTM 和 GRU 模型在一定程度上解决了 RNN 模型长期记忆的问题，但是对于一个长序列输入的情况下，记忆情况依然不乐观。在长序列输入的情况下，模型长期记忆经过反复的记忆、遗忘、记忆、遗忘、记忆的过程中，最初期的记忆依然会丢失。另外每当数据经过模型时，数据的信息便会丢失一些，随着层数的不断增加，最初的数据中包含的信息便会损失殆尽。为了解决这些问题，便提出了 Attention 机制和 ResNet（残差网络）。

---

# Attention 机制
在Seq2Seq模型中，一般使用两个RNN，一个作为编码器，一个作为解码器：编码器的作用是将输入数据编码成一个特征向量，然后解码器将这个特征向量解码成预测结果。当输入序列特别长的时候，LSTM模型和GRU模型依然会丢失早起时间数据的特征。对此，提出了一种新的将每个时间片的输出都提供给解码器的方式，那么接下来的提出的模型就是Attention[1][2]。
在这里，Attention是一个介于编码器和解码器之间的一个接口，用于将编码器的编码结果以一种更有效的方式传递给解码器。一个特别简单且有效的方式就是让解码器知道哪些特征重要，哪些特征不重要，即让解码器明白如何进行当前时间片的预测结果和输入编码的对齐，Attention模型学习了编码器和解码器的对齐方式，因此也被叫做对齐模型（Alignment Model）.
Attention有两种类型，一种是作用到编码器的全部时间片，这种Attention叫做全局（Global）Attention，另外一种值作用到时间片的一个子集，叫做局部（Local）Attention，这里要介绍的Attention都是全局的。
可以分解为以下几个步骤[3]:
![Attention.gif](https://cdn.nlark.com/yuque/0/2023/gif/741660/1679279297249-7cbd87d3-01e2-4e0d-b7ab-36641ed0045f.gif#averageHue=%23fefefe&clientId=ue20f455d-4537-4&from=drop&id=u7e8625dc&name=Attention.gif&originHeight=531&originWidth=700&originalType=binary&ratio=2&rotation=0&showTitle=false&size=186805&status=done&style=none&taskId=uff38546d-c4c8-41cd-902e-e3382c4f0f2&title=)
1. 生成编码节点
2. 为每个编码器的隐藏状态计算一个得分
3. 使用softmax 对得分进行归一化
4. 使用score 对隐藏状态进行加权
5. 对特征向量求和
6. 将特征向量应用的解码器

从广义上讲，注意力是网络架构的一个组成部分，负责管理和量化相互依赖性：

1. 输入和输出元素之间（普通 Attention 机制）
2. 在输入元素中（Self-Attention 机制）
## Example
Attention RNN: [https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)

1m 44s (- 24m 24s) (5000 6%) 2.8404
3m 29s (- 22m 43s) (10000 13%) 2.2784
5m 14s (- 20m 57s) (15000 20%) 1.9836
7m 3s (- 19m 23s) (20000 26%) 1.7378
8m 50s (- 17m 40s) (25000 33%) 1.5128
10m 32s (- 15m 49s) (30000 40%) 1.3423
12m 13s (- 13m 57s) (35000 46%) 1.1870
13m 53s (- 12m 9s) (40000 53%) 1.0951
15m 35s (- 10m 23s) (45000 60%) 0.9596
17m 16s (- 8m 38s) (50000 66%) 0.8616
18m 58s (- 6m 53s) (55000 73%) 0.7997
20m 37s (- 5m 9s) (60000 80%) 0.7232
22m 16s (- 3m 25s) (65000 86%) 0.6569
23m 54s (- 1m 42s) (70000 93%) 0.6022
25m 33s (- 0m 0s) (75000 100%) 0.5390

> vous etes la chef .
= you re the leader .
< you re the leader . <EOS>

> j ai bien peur que non .
= i m afraid not .
< i m afraid of <EOS>

> c est un ange .
= she s an angel .
< he s an office . <EOS>

> ils sont mes amis .
= they re my friends .
< they re my friends . <EOS>

> nous sommes tous en colere .
= we re all angry .
< we re all angry . <EOS>

> vous etes tres ouvert .
= you re very open .
< you re very open . <EOS>

> j attends impatiemment ta lettre .
= i am looking forward to your letter .
< i am looking forward to your letter . <EOS>

> vous etes tres avisees .
= you re very wise .
< you re very wise . <EOS>

> tu es tres effrontee .
= you re very forward .
< you re very forward . <EOS>

> vous n etes pas habillees .
= you re not dressed .
< you re not dressed . <EOS>
## Self-Attention
Transformer 中抛弃了传统的 CNN 和 RNN ，整个网络结构完全是由 Attention 机制组成。更准确地讲，Transformer 由且仅由 self-Attenion 和 Feed Forward Neural Network 组成。
假设，我们有输入 X，将输入 X 分别与三个不同的权值矩阵相乘，得到 Q，K，V 三个矩阵
$\begin{array}{ll}
X·W^Q = Q \\
X·W^K = K \\
X·W^V = V \\
\end{array}$
${Attention}(Q,K,V) - {softmax}(\frac{QK^T}{\sqrt{d_k}})V$
其中根据向量内积可以知道$QK^T$表征两个向量的夹角，表征一个向量在另一个向量上的投影。也就是计算向量相关性。
Attention整套流程可以分为三个部分：
第一是计算$\frac{QK^T}{\sqrt{d_k}}$，此处将得到各个单词与其它单词的关注矩阵
第二是softmax上述矩阵
第三则是将上述softmax的结果与V相乘
> While for small values of $d_k$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_k$[20]. We suspect that for large values of $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $1/\sqrt{d_k}$

有时候，点乘的数值会非常大，再经过 softmax 函数之后梯度会变得非常不明显[20]。因此选择乘$1/\sqrt{d_k}$，使得点乘的数据不失去数据分布特性的前提下增加梯度的明显程度。
句子如果没有达到句子的最大长度，则需要进行填充，如填充字符P，对应编码0。而0对应的嵌入向量并非为0向量（不清楚的可以回去位置编码那一小节的开头重新观看），那么在计算自注意力的时候，填充字符肯定也会参与计算并产生注意的权值。显然这是有问题的，我们并不需要去关注填充字符，换句话说，句子中的任何词，对填充字符的关注权重应该为0。

---

# 残差网络
在VGG中，卷积网络达到了19层，在GoogLeNet中，网络史无前例的达到了22层。那么，网络的精度会随着网络的层数增多而增多吗？在深度学习中，网络层数增多一般会伴着下面几个问题

1. 计算资源的消耗
2. 模型容易过拟合
3. 梯度消失/梯度爆炸问题的产生

随着网络层数的增加，网络发生了退化（degradation）的现象：随着网络层数的增多，训练集loss逐渐下降，然后趋于饱和，当你再增加网络深度的话，训练集loss反而会增大。注意这并不是过拟合，因为在过拟合中训练loss是一直减小的。
当网络退化时，浅层网络能够达到比深层网络更好的训练效果，这时如果我们把低层的特征传到高层，那么效果应该至少不比浅层的网络效果差，或者说如果一个VGG-100网络在第98层使用的是和VGG-16第14层一模一样的特征，那么VGG-100的效果应该会和VGG-16的效果相同。所以，我们可以在VGG-100的98层和14层之间添加一条直接映射（Identity Mapping）来达到此效果。
![ResNet.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679363699164-f924241d-7a78-467a-a1f9-bdaf79a58672.png#averageHue=%23f0f0ee&clientId=u407f7ae9-231d-4&from=drop&id=u5d10acd8&name=ResNet.png&originHeight=338&originWidth=160&originalType=binary&ratio=2&rotation=0&showTitle=false&size=13357&status=done&style=none&taskId=uf243c7c2-a583-4da4-ab09-0c90b3103b7&title=)
> 在统计学中，残差和误差是非常容易混淆的两个概念。误差是衡量观测值和真实值之间的差距，残差是指预测值和观测值之间的差距。

# Layer Normalization
相较于CNN常用的 Batch Normalization，NLP更倾向于使用 Layer Normalization [17]。
> BN的主要思想是:**在每一层的每一批数据(一个batch里的同一通道)上进行归一化**
> LN的主要思想是:**是在每一个样本(一个样本里的不同通道)上计算均值和方差**，而不是 BN 那种在批方向计算均值和方差.


> LN 的提出是为了解决应用 BN 时遇到的部分缺点[18]：
> ① 当 Batch size 小时不能使用 BN 
> ② BN 不能用在 RNN 中


---

# Transformer
![transformer_architecture.jpeg](https://cdn.nlark.com/yuque/0/2023/jpeg/741660/1679378813403-a810ffd3-dc20-489b-99c0-3ec16b4c3c78.jpeg#averageHue=%23eeedea&clientId=u407f7ae9-231d-4&from=drop&id=ua4d93026&name=transformer_architecture.jpeg&originHeight=916&originWidth=672&originalType=binary&ratio=2&rotation=0&showTitle=false&size=66814&status=done&style=none&taskId=u9633973b-09f0-4cd4-9d75-440dc5c1b26&title=)
![transformer_encoder_decoder.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679391973383-de59e9a6-3bed-4e98-9378-97e6346d5ec0.png#averageHue=%23dde8da&clientId=u3b642e81-51fe-4&from=drop&id=ub24b4dcb&name=transformer_encoder_decoder.png&originHeight=511&originWidth=720&originalType=binary&ratio=1&rotation=0&showTitle=false&size=16172&status=done&style=none&taskId=ua02654d9-9086-4ef6-8d45-6a462895d56&title=)

## 词向量编码
### TokenEmbedding
TokenEmbedding 采用 pytorch 自带的 nn.Embedding 函数进行向量编码，其为一个简单的存储固定大小的词典的嵌入向量的查找表，意思就是说，给一个编号，嵌入层就能返回这个编号对应的嵌入向量，嵌入向量反映了各个编号代表的符号之间的语义关系[19]。
> - **num_embeddings (**[int](https://docs.python.org/3/library/functions.html#int)**) – size of the dictionary of embeddings**
> - **embedding_dim (**[int](https://docs.python.org/3/library/functions.html#int)**) – the size of each embedding vector**

> 与其他序列转导模型类似，我们使用学习嵌入将输入令牌和输出令牌转换为维度$d_{model}$的向量。我们还使用通常学习的线性变换和softmax函数将解码器输出转换为预测的下一个令牌概率。在我们的模型中，我们在两个嵌入层和预softmax线性变换之间共享相同的权重矩阵，类似于[30]。在嵌入层中，我们将这些权重乘以$\sqrt{d_{model}}$

随着模型的训练，每个单词的向量也会随着模型参数的优化而发生改变，单词和单词之间的距离也会随之增加或者降低。距离越近的单词表示相关度越高，距离越远的单词表示相关度越低。
### PositionalEncoding
Positional Encoding 是 Facebook 在 2017 年发表的论文[22]提到的用于 CNN 模型进行时许数据训练的数据预处理方式。
> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence. To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks. 

由于 Transformer 模型输入不想 RNN 类模型输入时顺序输入，为了不丢失输入数据的位置信息，选择使用添加“positional encodings”的方式。为了给每个单词的位置编码，我们需要遵循以下原则：

1. 它应该为每个时间步长输出一个唯一的编码（单词在句子中的位置）
2. 在不同长度的句子中，任何两个时间步长之间的距离都应该一致。
3. 我们的模型应该毫不费力地推广到更长的句子。它的值应该是有界的。
4. 它必须是确定性的。

为了增加位置信息且不会因为位置信息过大而影响 token 的信息，采用 sin 和 cos 函数，为了使不同位置的位置信息有差异化，通过$1/10000^{2i/d_{model}}$调整每个维度的波形频率，使位置信息有差异化
$\begin{array}{ll}
{PE}_{(pos, 2i)} = {sin}({pos}/10000^{2i/d_{model}}) \\
{PE}_{(pos, 2i+1)} = {cos}({pos}/10000^{2i/d_{model}}) \\
\end{array}$
![PositionalEncoding_visual.png](https://cdn.nlark.com/yuque/0/2023/png/741660/1679489952206-7eb43cd6-d40a-4807-80ef-288de975a9f9.png#averageHue=%23458557&clientId=u23dbade8-b3e1-4&from=drop&id=u319795e7&name=PositionalEncoding_visual.png&originHeight=192&originWidth=839&originalType=binary&ratio=2&rotation=0&showTitle=false&size=14909&status=done&style=none&taskId=ua51be32a-ffb8-41a1-af74-c5213400f10&title=)The positional encoding matrix for n=10,000, d=512, sequence length=100
上图为 Postional Encoding 可视化之后的效果
其中$pos \in [0, {max\_len}]$，最后会输出一个 $[{max\_len}, d_{model}]$结构的矩阵。
在 Self-Attention 中计算 Scaled Dot-Product Attention 时我们可以发现，除了得到了两个向量的向量内积外，同时还可以得到每个向量之间的距离。
对于一个 2 * 2 大小的位置向量$PE_{pos}$任意的偏移量$\phi$，获得偏移后的向量$PE_{pos+\phi}$，只需要点乘转移矩阵$M_{(\phi, pos)}$[24]就可以获得
$M_{(\phi, pos)} = \begin{bmatrix}cos(\omega_k·\phi), sin(\omega_k·\phi) \\
-sin(\omega_k·\phi), cos(\omega_k·\phi)\end{bmatrix}$
采用正余弦的原因有很多，其中包括：

1. 不同频率的正余弦可以有效区分不同位置的点
2. 能够保持所有数据都在$[-1, 1]$之间特特性可以省去 normalization 的步骤。
## 模型结构
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer)
```python
encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
encoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout,
                                                    activation, layer_norm_eps, batch_first, norm_first,
                                                    **factory_kwargs)
decoder_norm = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
```
```python
memory = self.encoder(src, mask=src_mask, src_key_padding_mask=src_key_padding_mask)
output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask,
                              tgt_key_padding_mask=tgt_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
return output
```
Transformer 模型是典型的 Encoder - Decoder 模型结构，Encoder 输出的 memory 数据会作为 Decoder 输入的一部分全链接到每一层 Decoder，增强 memory 的记忆能力，即使 Decoder 层数增加也不会降低 memory 的性能（类似残差网络中直接映射的意义）。
## Encoder
```python
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.enable_nested_tensor = enable_nested_tensor
        self.mask_check = mask_check
```
```python
        output = src
        for mod in self.layers:
            output = mod(output, src_mask=mask, is_causal=is_causal, src_key_padding_mask=src_key_padding_mask_for_layers)

        if self.norm is not None:
            output = self.norm(output)

        return output
```

```python
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)


        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
```
```python
# 残差，add&norm
x = self.norm1(x + self._sa_block(x, src_mask, src_key_padding_mask))
# 残差，add&norm
x = self.norm2(x + self._ff_block(x))
```
### Self Attention
```python
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False)[0]
        return self.dropout1(x)
```
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention)
batch_first 默认 false
```python
self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
```
MultiheadAttention 类的实现方法：
```python
def forward(
            self,
            query: Tensor,
            key: Tensor,
            value: Tensor,
            key_padding_mask: Optional[Tensor] = None,
            need_weights: bool = True,
            attn_mask: Optional[Tensor] = None,
            average_attn_weights: bool = True,
            is_causal : bool = False) -> Tuple[Tensor, Optional[Tensor]]:
    attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, self.out_proj.weight, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight, k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
                average_attn_weights=average_attn_weights,
                is_causal=is_causal)
    return attn_output, attn_output_weights
```
其中 F.multi_head_attention_forward 最核心的内容就是实现部分如下（忽略掉所有参数检查部分）
```python
        attn_output = scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        # 重排列
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        # 卷积
        attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
        # 重排列
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if not is_batched:
            # squeeze the output if input was unbatched
            attn_output = attn_output.squeeze(1)
        return attn_output, None
```
scaled_dot_product_attention: [https://sensoro.yuque.com/ivn8up/yo30yr/gygqbasef9h45dt3#JY5Jk](#JY5Jk)
### Feed Forward
```python
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)
```
### 源代码
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoderLayer)
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerEncoder)
## Decoder
```python
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
```
```python
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, tgt_mask=tgt_mask,
                         memory_mask=memory_mask,
                         tgt_key_padding_mask=tgt_key_padding_mask,
                         memory_key_padding_mask=memory_key_padding_mask)


        if self.norm is not None:
            output = self.norm(output)


        return output
```

```python
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)


        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)
```
```python
# 残差，add&norm
x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
# 残差，add&norm
x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
# 残差，add&norm
x = self.norm3(x + self._ff_block(x))
```
### Self Attention
```python
    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)
```
### Multi-Head Attention
```python
    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)
```
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention](https://pytorch.org/docs/stable/_modules/torch/nn/modules/activation.html#MultiheadAttention)
这里需要注意的是第二层的 Multi-Head Attention 输入的 K，V 从原本的输入 X 变成了 Encoder 的输出 memory，这里不再是 Decoder 的输入的单词与单词之间的相关度，而变成了 Decoder 的输入与记忆中的每个单词的相关度。
![transformer_attention.jpeg](https://cdn.nlark.com/yuque/0/2023/jpeg/741660/1679391896040-dc0909ce-4a40-4b0c-8176-dc28a3916e45.jpeg#averageHue=%23faf2ec&clientId=u3b642e81-51fe-4&from=drop&id=uc998f608&name=transformer_attention.jpeg&originHeight=413&originWidth=437&originalType=binary&ratio=1&rotation=0&showTitle=false&size=37914&status=done&style=none&taskId=ub0cd6bc8-2452-4178-a202-3c55b923130&title=)
### Feed Forward
```python
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)
```
### 源代码
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoderLayer)
[https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder](https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#TransformerDecoder)
### Example
transformer translator：[https://pytorch.org/tutorials/beginner/translation_transformer.html](https://pytorch.org/tutorials/beginner/translation_transformer.html)

Epoch: 1, Train loss: 5.342, Val loss: 4.107, Epoch time = 243.991s
Epoch: 2, Train loss: 3.760, Val loss: 3.308, Epoch time = 241.590s
Epoch: 3, Train loss: 3.156, Val loss: 2.893, Epoch time = 240.769s
Epoch: 4, Train loss: 2.765, Val loss: 2.631, Epoch time = 243.130s
Epoch: 5, Train loss: 2.478, Val loss: 2.436, Epoch time = 242.140s
Epoch: 6, Train loss: 2.249, Val loss: 2.300, Epoch time = 250.141s
Epoch: 7, Train loss: 2.057, Val loss: 2.189, Epoch time = 239.043s
Epoch: 8, Train loss: 1.893, Val loss: 2.123, Epoch time = 240.658s
Epoch: 9, Train loss: 1.754, Val loss: 2.052, Epoch time = 237.791s
Epoch: 10, Train loss: 1.628, Val loss: 2.005, Epoch time = 240.043s
Epoch: 11, Train loss: 1.519, Val loss: 1.975, Epoch time = 247.284s
Epoch: 12, Train loss: 1.417, Val loss: 1.956, Epoch time = 240.153s
Epoch: 13, Train loss: 1.331, Val loss: 1.965, Epoch time = 247.078s
Epoch: 14, Train loss: 1.249, Val loss: 1.960, Epoch time = 245.977s
Epoch: 15, Train loss: 1.171, Val loss: 1.915, Epoch time = 245.422s
Epoch: 16, Train loss: 1.100, Val loss: 1.916, Epoch time = 252.000s
Epoch: 17, Train loss: 1.036, Val loss: 1.913, Epoch time = 248.728s
Epoch: 18, Train loss: 0.976, Val loss: 1.922, Epoch time = 254.254s
A group of people stand in front of an igloo . 

---

# GPT
## **GPT-1[7]**
## **GPT-2[8]**
## **GPT-3[9]**
## 参考文档
[1] Vaswani A ,  Shazeer N ,  Parmar N , et al. Attention Is All You Need[J]. arXiv, 2017.
[2] [https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3#0458](https://towardsdatascience.com/attn-illustrated-attention-5ec4ad276ee3#0458)
[3] [https://zhuanlan.zhihu.com/p/342235515](https://zhuanlan.zhihu.com/p/342235515)
[4] [https://zhuanlan.zhihu.com/p/48508221](https://zhuanlan.zhihu.com/p/48508221)
[5] [https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html#torch.nn.RNN)
[6] [https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)
[7] Radford, A., Narasimhan, K., Salimans, T. and Sutskever, I., 2018. Improving language understanding by generative pre-training.
[8] Radford, A., Wu, J., Child, R., Luan, D., Amodei, D. and Sutskever, I., 2019. Language models are unsupervised multitask learners. *OpenAI blog*, *1*(8), p.9.
[9] Brown, Tom B., Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan et al. “Language models are few-shot learners.” *arXiv preprint arXiv:2005.14165* (2020).
[10] [https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU)
[11] [http://colah.github.io/posts/2015-08-Understanding-LSTMs/](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
[12] Rumelhart D E , Hinton G E , Williams R J . Learning representations by back-propagating errors. 1988.
[13] [https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf](https://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf)
[14] [https://ai.dinfo.unifi.it/paolo/ps/tnn-94-gradient.pdf](https://ai.dinfo.unifi.it/paolo/ps/tnn-94-gradient.pdf)
[15] [https://zhuanlan.zhihu.com/p/32085405](https://zhuanlan.zhihu.com/p/32085405)
[16] [https://zhuanlan.zhihu.com/p/32481747](https://zhuanlan.zhihu.com/p/32481747)
[17] Xu J , Sun X , Zhang Z , et al. Understanding and Improving Layer Normalization[J]. 2019.
[18] [https://zhuanlan.zhihu.com/p/568938529](https://zhuanlan.zhihu.com/p/568938529)
[19] [https://www.jianshu.com/p/63e7acc5e890](https://www.jianshu.com/p/63e7acc5e890)
[20] Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. _CoRR_, abs/1703.03906, 2017. 
[21] [https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model](https://datascience.stackexchange.com/questions/51065/what-is-the-positional-encoding-in-the-transformer-model)
[22] Gehring J , Auli M , Grangier D , et al. Convolutional Sequence to Sequence Learning[J]. 2017.
[23] [https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/](https://machinelearningmastery.com/a-gentle-introduction-to-positional-encoding-in-transformer-models-part-1/)
[24] [https://kazemnejad.com/blog/transformer_architecture_positional_encoding/](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/)

