# 作业题2：基于神经网络的文本分类器实现

## 作业说明
1. **model.py:** 需要实现的部分。它使用[PyTorch](https://github.com/pytorch/pytorch)实现了一个非常基本的神经网络模型。提供了一些代码，但不包括重要的功能。请实现[深度平均网络 (Deep Averaging Network, DAN)](https://www.aclweb.org/anthology/P15-1162.pdf)进行文本分类。你可以任意修改，使它成为一个更好的模型。但是，原始版本的`DanModel`也将被测试，结果将用于评分，所以原始的`DanModel`也必须与你的`model.py`实现一起运行。
2. **main.py:** 文本分类任务的训练代码。
3. **setup.py:** 此文件目前空白，但是如果你的分类器实现需要做一些数据下载（例如预训练的词嵌入），你可以在这里实现。它将在运行model.py的实现之前运行。
4. **data/:** 对应数据集，来自Stanford Sentiment Treebank，见参考文献。

## 作业细节

重要提示:
- 下面[代码结构](#代码结构)包含详细描述代码内容，包括您需要实现的部分的描述。
- 本代码唯一允许的外部库是`numpy`和`pytorch`，不允许其他外部库。由于数据集较小，使用CPU可以在几分钟（<30分钟）内训练出与原始论文中相似大小的DAN模型，同时也鼓励训练可能需要GPU的更高级的模型。请查看可用的资源，如[魔搭](https://www.modelscope.cn/)或[谷歌的Colab](https://colab.research.google.com/)。
- 我们将使用以下命令（即'run_exp.sh'）运行您的代码，同时使用原始的'main.py'和更新的'model.py'，如果您在那里做了任何修改。因此，请确保您认为最好的设置可以使用以下命令（其中将“CAMPUSID”替换为您的校园ID）：

- `CAMPUSID = "202xxxxx”`
- `mkdir -p CAMPUSID`
- `python main.py——train=data/sst-train.txt——dev=data/sst-dev.txt——test=data/sst-test.txt——dev_out=$CAMPUSID/sst-dev-output.txt`

-请记住在你自己的`run_exp.sh`中设置默认的超参数设置，因为我们也会通过`bash run_exp.sh`来运行你的实验（没有额外的参数）。
-引用精度：如果你完全按照我们的方式实现，并使用默认的超参数，并使用相同的环境（python 3.8 + numpy 1.21.1 + pytorch 1.10.2），你可能会得到dev=0.3951, test=0.4122的精度。

提交文件应该是一个zip文件，结构如下（假设校园id为‘ CAMPUSID ’）：

- CAMPUSID/
- CAMPUSID/main.py `# completed main.py(即函数'pad_sentences()')`
- CAMPUSID/model.py `# 完成model.py`
- CAMPUSID/vocabulary.py `# 无需改动`
- CAMPUSID/setup.py `# 仅当你需要设置其他东西时修改`
- CAMPUSID/SST -dev-output.txt `# SST数据的dev set输出`
- CAMPUSID/SST -test-output.txt `# SST数据测试集的输出`
- CAMPUSID/report.pdf `#（可选），report。此文件你可以描述任何你做过的特别新颖或有趣的事情。`
- CAMPUSID/README `#（可选）仅当你使用预先训练好的词向量，如GloVE和FastText。不要上传词向量(word embedding)文件。相反，请在README中注明您在main.py中用于“——emb_file”的词向量文件的下载链接。`

评分标准:
- **100:** 提交的内容实现了一些新的东西，并实现了特别大的精度提高(对于SST，这将在基线上提高2%左右)
- **95:** 你在缺失的部分上额外实现一些其他的东西，一些例子包括：
-改变训练程序，如小批处理，优化器，提前停止，学习率调度。
-结合预训练的词向量，如那些从[fasttext](https://fasttext.cc/)或[GloVe](https://nlp.stanford.edu/projects/glove/)
-显著改变模型架构
- **90:** 你实现了所有缺失的部分，原始的`model.py`代码达到了与我们的参考实现相当的精度（在SST上约为40%）
- **85:** 所有缺失的部分都实现了，但精度无法与参考相比。
- **80或以下：** 部分缺失的部件不实现。

工具:
- `prepare_submit.py`: 可以帮助创建(1)或检查(2)要提交的zip文件。如果格式不符合预期，它将抛出断言错误，作业*不接受未通过此检查的提交*。用法: (1)用你的输出创建并检查一个zip文件，运行`python prepare_submit.py path/to/your/output/dir CAMPUSID`，(2)检查你的zip文件，运行`python3 prepare_submit.py path/to/your/submit/zip/file.zip CAMPUSID`

## 代码结构
下面是这个repo中主要组件的步骤详解。请注意，有些函数没有完全实现，其中**要实现的**部分将调用`raise NotImplementedError()`。

### [model.py](model.py)
这个文件包含一个`BaseModel`类和一个`DanModel`类。以下是要实现的部分：
—**define_model_parameters()**：定义模型的参数，如嵌入层、前馈层、激活函数（ReLU）等。参见[PyTorch API](https://pytorch.org/docs/stable/nn.html)了解不同的层。
- **init_model_parameters()**：使用统一的初始化方法初始化模型的参数。参考`Understanding the difficulty of training deep feedforward neural networks - Glorot, X. & Bengio, Y. (2010)`关于Xavier/Glorot的初始化和[more details about initialization in general](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79).
- **load_embedding()**： 逐行读取文件，为出现在`vocab`中的单词构建一个词嵌入矩阵（numpy.array）。
- **copy_embedding_from_numpy**：将词嵌入(numpy.array)复制到PyTorch的嵌入矩阵中。

### [main.py](main.py)
这个文件包含学习文本分类器的训练和评估函数。只有一个函数**需要实现**：
- **pad_sentences()**：给定一个小批量不同长度的句子（即单词id列表的列表）（例如，`[[1,2,5],[3,4],[4,6,8,9]]`表示一个由3个句子组成的小批量，其中它们的单词长度不同`|s_1|=3, |s_2|=2, |s_3|=4`），找到最大的单词序列长度（即`max_seq_length=4`），并将pad id添加到句子的末尾，形成一个大小为`[batch_size, max_seq_length]`的小批量（例如，在本例中`[3,4]`）。

### [vocab.py](vocab.py)
该文件读取标记化的句子列表，并为单词构建词汇表。我们可以重用它来构建标签的词汇表（例如，情感分类任务中的`积极`和`消极`）。

## 参考文献

Stanford Sentiment Treebank: https://www.aclweb.org/anthology/D13-1170.pdf