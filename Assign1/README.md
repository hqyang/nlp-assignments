# 作业题1：古腾堡计划文本预处理工具实现

## 一、作业背景
古腾堡计划（Project Gutenberg）收录了大量中英文经典书籍的电子化文本，是自然语言处理的重要语料来源。

本次作业要求实现一套文本预处理工具，对从古腾堡计划获取的中英文文本进行标准化处理（如清洗、规范化等），为后续的分词、特征提取等任务奠定基础（参考代码：[02-tokenization/text_preprocessing.ipynb](https://github.com/hqyang/nlp-codes/tree/main/02-tokenization/text_preprocessing.ipynb)）。

## 二、作业说明
### 1. 数据获取 (已提供)
实现fetch_gutenberg_text函数，从指定本地路径读取古腾堡文本文件（需处理文件编码及元数据）。

函数定义：def fetch_gutenberg_text(file_path='./data/1342-0.txt')

处理逻辑：需自动识别文件编码（优先utf-8，失败则尝试latin-1），并提取“*** START OF THE PROJECT GUTENBERG EBOOK ... ***"与"*** END OF THE PROJECT GUTENBERG EBOOK ... ***"标记之间的正文内容。

验证数据：提供3个验证文件，路径为data/preprocessing/1342-0.txt、24264-0.txt、23950-0.txt。

原文件链接: 
- 英文文本：《傲慢与偏见》（Jane Austen, Pride and Prejudice）
    链接：https://www.gutenberg.org/files/1342/1342-0.txt 
- 中文文本：
    1. 《红楼梦》（曹雪芹, A Dream Of Red Mansions）
    链接: https://www.gutenberg.org/files/24264/24264-0.txt
    2. 《三国演义》(罗贯中) 
    链接: https://www.gutenberg.org/files/23950/23950-0.txt

### 2. 预处理功能实现 (待实现)
实现`normalize_doc`函数及相关辅助函数，支持以下可配置的预处理步骤（通过参数控制开关），最终返回预处理后的文本及语言类型（`'en'`/`'zh'`/`'unknown'`）。

| 预处理步骤 | 函数要求 | 说明 |
|------------|----------|------|
| 移除HTML标签 | `strip_html_tags(text)` | 移除`<iframe>`、`<script>`等标签，标准化换行符为`\n` |
| 扩展英文缩写 | 调用`contractions.fix(text)` | 将`don't`转为`do not`等 |
| 移除重音字符 | `remove_accented_chars(text)` | 将`café`转为`cafe`等 |
| 文本规范化 | - | 英文：转为小写；中文：繁体转简体（使用`OpenCC('t2s')`） |
| 词形还原 | `pos_tag_wordnet(tagged_tokens)`、`lemmatize_text(text)` | 英文：基于WordNet还原（如`running`→`run`）；中文：保留原词 |
| 移除特殊字符 | `remove_special_characters(text, remove_digits=remove_digits)` | 可选是否保留数字（英文保留字母/中文保留汉字） |
| 移除停用词 | `remove_stopwords(text, is_lower_case=text_lower_case)` | 英文用NLTK英文停用词表，中文用NLTK中文停用词表 |


### 3. 结果存储 (已提供)
将预处理后的文本保存至对应文件，命名规则为：原文件名为`xxx.txt`，处理后文件名为`xxx-p.txt`（如`./data/1342-0.txt`→`./data/1342-0-p.txt`）。

### 4. 结果输出与验证 (已提供)

当前文件夹目录如下: 
```
nlp-assignments/
├── data/
│   ├── 1342-0.txt
│   ├── 23950-0.txt
│   ├── 24264-0.txt
│   ├── 8001.txt
│   ├── 7337-0.txt
│   ├── 8001-p.txt          # 提供的处理后的示例文本
│   └── 7337-0-p.txt        # 提供的处理后的示例文本
└── Assign1/
    ├── README.md           # 作业说明
    ├── Assign1.py          # 不能改变此文件代码
    ├── Assign1_func.py     # 实现pass对应部分代码
    ├── prepare_submit.py   # 提交前必须使用，用于检查提交文件是否完全满足要求
    ├── requirements.txt    # 需要的安装包
    ├── sample.json         # 示例配置文件
    ├── sample_out.md       # 示例输出结果
    └── util.py             # 无需更改此代码 (被Assign1.py或Assign1_func.py调用)
```

你可以用下面的命令生成作业环境并安装需要的包
```
conda create -n nlp-fall25-assign1 python=3.10
conda activate nlp-fall25-assign1
pip install -r requirements.txt
```

当你完成Assign1-1.py后，可以在相应目录下运行下述的命令，获得sample_out.md的结果作为输出。通过对比，你可以确定你的代码是否正确实现。
```
python Assign1-1.py sample.json  > sample_out.md
```

输出文件sample_out.md的最后部分会显示 (“✅ 所有检查通过”: 说明所有检查已通过)
```
## 检查预处理结果：《Book01_Genesis》
✅ 所有检查通过

...

## 检查预处理结果：《Lau-zi dao de jing》
✅ 所有检查通过
```

## 四、提交代码说明
请将你的作业压缩成SID-Assign1.zip, 其中SID是你的学号(此目的是方便老师用自动脚本评分)。解压后文件内容如下: 
```
SID-Assign1/
├── Assign1.py          # 原文件
├── Assign1_func.py     # 实现后的代码
├── out.md              # 运行Assign1.py得出的结果
└── util.py             # 原文件
````
Tools:

- prepare_submit.py can help to create(1) or check(2) the to-be-submitted zip file. It will throw assertion errors if the format is not expected, and we will not accept submissions that fail this check. Usage: 
  - (1) To create and check a zip file with your outputs, run python prepare_submit.py path/to/your/output/dir SID, 
  - (2) To check your zip file, run python prepare_submit.py path/to/your/submit/zip/SID-Assign1.zip SID

# 作业提交时间与评分标准
## 截止时间
- 2025年10月19日(北京时间)23:59

## 评分标准
|  内容 | 得分 |
|------------|------|
| 完全正确 | 100 |
| 正确完成normalize_doc的每个预处理步骤的调用 | 每个5分 (共35分 ) |
| 正确完成所有预处理步骤的函数 | 每个5分(共35分) |
| 文件的准确输出(out.md)  | 每个10分 (共30分) |

* **注意:** 
  - 允许3天迟交；往后按每天扣减1%总分