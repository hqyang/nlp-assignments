# 作业题1：古腾堡计划文本预处理工具实现

## 一、作业背景
古腾堡计划（Project Gutenberg）收录了大量中英文经典书籍的电子化文本，是自然语言处理的重要语料来源。本次作业要求实现一套文本预处理工具，对从古腾堡计划获取的中英文文本进行标准化处理（如清洗、规范化等），为后续的分词、特征提取等任务奠定基础（参考代码：02-tokenization/text_preprocessing.ipynb）。

## 二、作业说明
### 1. 数据获取 (已提供)
实现fetch_gutenberg_text函数，从指定本地路径读取古腾堡文本文件（需处理文件编码及元数据）。
函数定义：def fetch_gutenberg_text(file_path='./data/1342-0.txt')
处理逻辑：需自动识别文件编码（优先utf-8，失败则尝试latin-1），并提取“*** START OF THE PROJECT GUTENBERG EBOOK ... ***"与"*** END OF THE PROJECT GUTENBERG EBOOK ... ***"标记之间的正文内容。
测试数据：提供3个测试文件，路径为data/preprocessing/1342-0.txt、24264-0.txt、23950-0.txt（Github 仓库同步提供）。

参考原文件链接: 
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
将预处理后的文本保存至对应文件，命名规则为：原文件名为`xxx.txt`，处理后文件名为`xxx-p.txt`（如`./data/preprocessing/1342-0.txt`→`./data/preprocessing/1342-0-p.txt`）。

### 4. 结果输出与验证 (已提供)

当前文件夹目录如下: 
nlp-assignments/
|____ Assign1-1.md       # 作业说明  
|____ data/              # 数据子目录
| |____ 1342-0.txt       # 《傲慢与偏见》           
| |____ 23950-0.txt      # 《三国演义》  
| |____ 24264-0.txt      # 《红楼梦》   
| |____ 8001.txt         # 《Book01_Genesis》: 运行示例   
| |____ 7337-0.txt       # 《Lau-zi dao de jing》: 运行示例      
| |____ 8001-p.txt       # 《Book01_Genesis》: 处理后文本示例  
| |____ 7337-0.txt       # 《Lau-zi dao de jing》: 处理后文本示例 
|____ Assign1            # 作业目录 
| |____ Assign1-1.py     # 主程序 (需完成pass的代码实现)
| |____ sample.json      # 测试示例
| |____ sample_out.md    # 测试结果示例

你可以用下面的命令生成作业环境
```
conda create -n nlp-fall25-assign1 python=3.10
conda activate nlp-fall25-assign1
···

当你完成Assign1-1.py后，可以在相应目录下运行下述的命令，获得sample_out.md的结果作为输出。通过对比，你可以确定你的代码是否正确实现。
```
python Assign1-1.py sample.json  > sample_out.md
```

输出文件sample_out.md的最后部分会显示 (“✅ 所有检查通过”: 说明所有检查已通过)
···
## 检查预处理结果：《Book01_Genesis》
✅ 所有检查通过

...

## 检查预处理结果：《Lau-zi dao de jing》
✅ 所有检查通过
···

## 四、提交代码说明
请将你的作业压缩成SID-Assign1.zip, 其中SID是你的学号。解压后文件内容如下: 
SID-Assign1/
|__ Assign1-1.py        # 主程序 (需完成pass的代码实现)