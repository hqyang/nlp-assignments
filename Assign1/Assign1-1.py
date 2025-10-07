# -*- coding: utf-8 -*-
# @Author: Haiqin Yang
# @Date:   2025-10-05 18:20:05
# @Last Modified by:   Haiqin Yang
# @Last Modified time: 2025-10-07 15:37:54

#!/usr/bin/env python3

import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

from nltk.stem import PorterStemmer

from nltk.corpus import wordnet, stopwords

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.probability import FreqDist
 
import requests
from urllib.parse import urlparse

import re
from bs4 import BeautifulSoup

import contractions

import unicodedata

from opencc import OpenCC
import jieba
cc_zh = OpenCC('t2s')  # 中文繁体转简体

from tqdm import tqdm  # 导入tqdm库
import argparse
import json
from pathlib import Path  # 用于路径验证

def detect_language(text):
    '''
    更严谨的实现是使用langdetect包，但该包有时会误判中英混合文本为其他语言
    需要安装：pip install langdetect
    from langdetect import detect, LangDetectException

    def detect_language(text):
        if not text.strip():
            return 'unknown'
        try:
            lang = detect(text)
            # 映射为中文（zh）或英文（en），其他语言返回unknown
            return 'zh' if lang == 'zh-cn' else 'en' if lang == 'en' else 'unknown'
        except LangDetectException:
            return 'unknown'
    '''
    if not text.strip():
        return 'unknown'
    
    # 提取中文CJK字符和英文单词字符
    cjk_chars = re.findall(r'[\u4e00-\u9fff]', text)
    en_chars = re.findall(r'[a-zA-Z]', text)  # 英文特征：字母
    
    total = len(text.strip())
    if total == 0:
        return 'unknown'
    
    cjk_ratio = len(cjk_chars) / total
    en_ratio = len(en_chars) / total
    
    # 中文判定：CJK占比超30%
    if cjk_ratio > 0.3:
        return 'zh'
    # 英文判定：英文占比超30%，且CJK占比极低（避免中英混合文本误判）
    elif en_ratio > 0.3 and cjk_ratio < 0.1:
        return 'en'
    # 其他情况（如中英混合、其他语言）
    else:
        return 'unknown'

def get_statistics(text, n=10, k=10, lang='en'):
    '''
    获取文本中前n个高频词和长度最长的k个词
    
    参数：
        text: 输入文本
        n: 高频词数量（默认10）
        k: 最长词数量（默认10）
        lang: 语言（'en'英文/'zh'中文，默认'en'）
    
    返回：
        元组 (top_n_words, longest_k_words)，其中：
            - top_n_words: 列表，元素为 (词, 频率) 元组，按频率降序排列
            - longest_k_words: 列表，元素为 (词, 长度) 元组，按长度降序排列（长度相同则按词本身排序）
    '''    
    if not text.strip():
        return ([], [])  # 空文本返回两个空列表
    
    # 1. 分词（中英文适配）
    if lang == 'en':
        tokens = nltk.word_tokenize(text)
    elif lang == 'zh':
        tokens = jieba.lcut(text)  # 中文精确分词
    else:
        raise ValueError("Unsupported language. Use 'en' or 'zh'.")
    
    # 过滤空字符串（移除纯空格/空的token）
    tokens = [token.strip() for token in tokens]
    tokens = [token for token in tokens if token]  # 保留非空token
    
    # 2. 计算前n个高频词（复用原逻辑）
    fdist = FreqDist(tokens)
    top_n_words = fdist.most_common(n)  # 格式：[(词1, 频率1), (词2, 频率2), ...]
    
    # 3. 计算长度最长的k个词（去重后按长度排序）
    unique_tokens = list(set(tokens))  # 去重，避免重复词占据位置
    # 按长度降序排序（长度相同则按词的字母/字符顺序排序，保证稳定性）
    sorted_by_length = sorted(
        unique_tokens,
        key=lambda x: (-len(x), x)  # 先按长度的负数（降序），再按词本身（升序）
    )
    # 取前k个，并添加长度信息
    longest_k_words = [(word, len(word)) for word in sorted_by_length[:k]]
    
    return (top_n_words, longest_k_words)

def strip_html_tags(text):
    """
    移除文本中的HTML标签并提取纯文本内容，同时标准化换行符。

    功能说明：
        1. 使用BeautifulSoup解析HTML文本，移除`<iframe>`和`<script>`标签（通常包含非文本内容）
        2. 提取HTML中的纯文本内容
        3. 将各种换行符（\r、\n、\r\n）统一替换为单个\n，确保换行格式一致

    参数：
        text (str)：包含HTML标签的原始文本字符串

    返回：
        str：移除所有HTML标签后的纯文本，换行符已标准化为\n

    依赖：
        需要安装BeautifulSoup库（pip install beautifulsoup4）

    示例：
        >>> raw_html = "<p>Hello<br>World!</p><script>alert('test')</script>"
        >>> strip_html_tags(raw_html)
        'Hello\nWorld!'
    """
    pass
    
def remove_accented_chars(text):
    """
    移除文本中的重音字符（如é、ñ、ü等），将其转换为无重音的基础字符。
    
    处理逻辑：
        1. 使用`unicodedata.normalize('NFKD', text)`对文本进行Unicode规范化：
           - NFKD（Compatibility Decomposition, Canonical Composition）会将重音字符分解为
             基础字符 + 重音符号（例如：'é' 分解为 'e' + 重音符号）。
        2. 通过`.encode('ascii', 'ignore')`将规范化后的文本编码为ASCII：
           - ASCII编码不支持重音符号，`ignore`参数会忽略无法编码的重音符号部分，仅保留基础字符。
        3. 再通过`.decode('utf-8', 'ignore')`将字节流解码回UTF-8字符串，得到无重音的结果。
    
    参数：
        text (str)：包含重音字符的原始文本
    
    返回：
        str：移除重音后的文本（仅包含ASCII可表示的字符）
    
    示例：
        >>> remove_accented_chars("café cliché naïve")
        'cafe cliche naive'
        >>> remove_accented_chars("àèìòù ÁÉÍÓÚ ñ ç")
        'aeiou AEIOU n c'
    """
    pass
    
def pos_tag_wordnet(tagged_tokens):
    """
    将NLTK词性标注结果转换为WordNet词形还原器（Lemmatizer）兼容的词性标签。
    
    背景：
        NLTK的`pos_tag`函数返回的词性标签遵循Penn Treebank格式（如形容词为'JJ'、动词为'VB'等），
        而WordNet的`lemmatize`方法需要特定的词性标签（如形容词为`wordnet.ADJ`、动词为`wordnet.VERB`等），
        因此需要通过首字母映射实现格式转换。
    
    参数：
        tagged_tokens (list)：由`nltk.pos_tag`返回的词性标注列表，每个元素为元组`(word, tag)`，
                             其中`word`是词语，`tag`是Penn Treebank格式的词性标签（如'JJ'、'VB'）。
    
    返回：
        list：转换后的词性标注列表，每个元素为元组`(word, wordnet_tag)`，
              其中`wordnet_tag`是WordNet兼容的词性标签（如`wordnet.ADJ`、`wordnet.VERB`）。
              若标签无法匹配，默认映射为名词（`wordnet.NOUN`）。
    
    映射规则：
        - 'j'（形容词，如Penn标签'JJ'）→ `wordnet.ADJ`
        - 'v'（动词，如Penn标签'VB'）→ `wordnet.VERB`
        - 'n'（名词，如Penn标签'NN'）→ `wordnet.NOUN`
        - 'r'（副词，如Penn标签'RB'）→ `wordnet.ADV`
    
    示例：
        >>> from nltk import pos_tag, word_tokenize
        >>> tagged = pos_tag(word_tokenize("running fast"))  # [('running', 'VBG'), ('fast', 'RB')]
        >>> pos_tag_wordnet(tagged)
        [('running', wordnet.VERB), ('fast', wordnet.ADV)]
    """
    pass


def lemmatize_text(text):
    """
    对文本进行词形还原（Lemmatization），将词语还原为其基本形式（如动词第三人称→原形、名词复数→单数）。
    
    处理流程：
        1. 分词：将文本拆分为独立词语（使用NLTK的`word_tokenize`）。
        2. 词性标注：为每个词语添加Penn Treebank格式的词性标签（使用NLTK的`pos_tag`）。
        3. 标签转换：将词性标签转换为WordNet兼容格式（调用`pos_tag_wordnet`）。
        4. 词形还原：使用WordNet词形还原器，根据转换后的词性标签对每个词语进行还原。
        5. 拼接：将还原后的词语重新拼接为文本。
    
    参数：
        text (str)：待处理的原始文本（英文）。
    
    返回：
        str：经过词形还原后的文本，词语均为基本形式。
    
    依赖：
        - 需要加载NLTK的`punkt`分词模型和`averaged_perceptron_tagger`词性标注模型（可通过`nltk.download()`下载）。
        - 需要初始化WordNet词形还原器（如`wnl = WordNetLemmatizer()`）。
    
    示例：
        >>> wnl = WordNetLemmatizer()  # 假设已初始化
        >>> lemmatize_text("Cats are running quickly")
        'cat be run quick'
    """
    pass       

def remove_special_characters(text, lang='en', remove_digits=False):
    """
    移除文本中的特殊字符（如标点符号、符号等），可选择是否保留数字。
    
    处理逻辑：
        1. 根据`remove_digits`参数构建正则匹配模式:
           - 当`remove_digits=False`（默认）:
             - lang='en': 保留字母(a-zA-Z)、数字(0-9)和空格(\s)，移除其他不相关字符。
             - lang='zh': 保留中文(一-龥)、数字(0-9)和空格(\s)，移除其他不相关字符。
           - 当`remove_digits=True`:
             - lang='en': 保留字母(a-zA-Z)、数字(0-9)和空格(\s)，移除其他不相关字符。
             - lang='zh': 保留中文(一-龥)、数字(0-9)和空格(\s)，移除其他不相关字符。
        2. 使用`re.sub`将匹配到的特殊字符替换为空字符串，实现移除效果。
        3. 额外处理：移除多余的换行符和空白字符，确保文本整洁。
    
    参数：
        text (str)：待处理的原始文本。
        remove_digits (bool)：是否移除数字，默认False（保留数字）。
        remove_special_characters (bool): 是否移除特殊字符，默认False（保留特殊字符）
    
    返回：
        str：移除特殊字符后的文本（仅保留指定允许的字符）。
    
    示例：
        >>> remove_special_characters("Hello, 世界! 123 🙂🙂🙂", lang='en')
        'Hello  123'
        
        >>> remove_special_characters("Hello, 世界! 123 🙂🙂🙂", lang='zh')
        '世界 123'
        
        >>> remove_special_characters("Hello, 世界! 123 🙂🙂🙂", lang='en', remove_digits=True)
        'Hello '
        
        >>> remove_special_characters("Hello, 世界! 123 🙂🙂🙂", lang='zh', remove_digits=True)
        '世界'    
    """
    pass

def remove_stopwords(text, is_lower_case=False, stopwords=None, lang='en'):
    """
    移除文本中的停用词（如英文的"the"、"is"，中文的"的"、"了"等无实际语义的高频词），支持中英文。
    
    处理流程：
        1. 加载停用词表：若未提供自定义停用词表（stopwords），则根据语言（lang）加载默认停用词表：
           - 英文：使用NLTK的英文停用词表（nltk.corpus.stopwords.words('english')）。
           - 中文：使用NLTK的中文停用词表（nltk.corpus.stopwords.words('chinese')）。
        2. 分词：根据语言选择分词工具：
           - 英文：使用NLTK的`word_tokenize`进行分词。
           - 中文：使用Jieba的`lcut`（精确模式）进行分词。
        3. 过滤停用词：根据`is_lower_case`判断是否需要将词语小写后再匹配停用词表，保留非停用词。
        4. 拼接：将过滤后的词语重新拼接为文本。
    
    参数：
        text (str)：待处理的原始文本。
        is_lower_case (bool)：文本是否已转为小写，默认False（需将词语小写后再匹配停用词）。
        stopwords (list, optional)：自定义停用词表，若为None则使用默认表。
        lang (str)：语言类型，'en'（英文）或'zh'（中文），默认'en'。
    
    返回：
        str：切词并移除停用词后用空格相连的文本。
    
    依赖：
        - 英文：需加载NLTK的`punkt`分词模型（nltk.download('punkt')）和英文停用词表（nltk.download('stopwords')）。
        - 中文：需安装Jieba（pip install jieba）和加载NLTK的中文停用词表（nltk.download('stopwords')）。
    
    示例：
        >>> # 英文示例
        >>> remove_stopwords("The quick brown fox jumps over the lazy dog", lang='en')
        'quick brown fox jumps lazy dog'
        >>> # 中文示例
        >>> remove_stopwords("这只敏捷的棕色狐狸跳过了那只懒狗", lang='zh')
        '只 敏捷 棕色 狐狸 跳过 只 懒 狗'
    """
    pass    
 
def normalize_doc(doc, 
                     html_stripping=True, 
                     contraction_expansion=True,
                     accented_char_removal=True, 
                     text_lower_case=True,
                     text_lemmatization=True, 
                     special_char_removal=True,
                     stopword_removal=True, 
                     remove_digits=False, # Default: keep digits
                     zh_simplification=True,
                     isDebug=False):
    """
    规范化文本语料库，支持多种预处理操作，包括HTML标签移除、缩写扩展、重音字符移除、
    小写转换、词形还原、特殊字符移除、停用词移除等，适用于中英文文本。
    参数：
        doc (str): 输入文档。
        html_stripping (bool): 是否移除HTML标签，默认True。
        contraction_expansion (bool): 是否扩展缩写，默认True。
        accented_char_removal (bool): 是否移除重音字符，默认True。
        text_lower_case (bool): 是否将文本转换为小写，默认True。
        text_lemmatization (bool): 是否进行词形还原，默认True。
        special_char_removal (bool): 是否移除特殊字符，默认True。
        stopword_removal (bool): 是否移除停用词，默认True。
        remove_digits (bool): 是否移除数字，默认False。
        zh_simplification (bool): 是否将中文繁体转换为简体，默认True。
        isDebug (bool): 是否打印调试信息，默认False。
    返回：
        元组 `(doc, lang)`，其中：
            - doc (str)：预处理后的文本(切词后用空格相连)
            - lang (str)：文本语言（'en'/'zh'/'unknown'）    
    """                              
    pass


def fetch_gutenberg_text(file_path=None):
    """
    读取本地古腾堡txt文件，提取正文内容（去除首尾元数据标记）
    
    参数：
        file_path (str): 本地txt文件路径，默认 './data/1342-0.txt'
    
    返回：
        str: 提取的正文内容；若文件读取失败/无有效标记，返回None
    
    示例：
        >>> content = fetch_gutenberg_text('./data/1342-0.txt')
        >>> print(content[:500])  # 打印正文前500字符
           
    说明: 
        你可以自己实现输入 fetch_gutenberg_text(link='https://www.gutenberg.org/files/1342/1342-0.txt'):
        >>> content = fetch_gutenberg_text('https://www.gutenberg.org/files/1342/1342-0.txt')
        >>> print(content[:500])  # 打印正文前500字符
    """
    # 1. 验证文件格式（仅支持txt）
    if not file_path.endswith('.txt'):
        print(f"错误：{file_path} 不是.txt格式文件，仅支持文本文件")
        return None

    # 2. 正则匹配古腾堡标准标记（提取正文核心）
    # 匹配 "*** START OF THE PROJECT GUTENBERG EBOOK ... ***" 及变体
    start_pat = re.compile(
        r'\*{3}\s+START OF (?:THE|THIS)?\s+PROJECT GUTENBERG EBOOK.*?\*{3}',
        re.IGNORECASE | re.DOTALL
    )
    # 匹配 "*** END OF THE PROJECT GUTENBERG EBOOK ... ***" 及变体
    end_pat = re.compile(
        r'\*{3}\s+END OF (?:THE|THIS)?\s+PROJECT GUTENBERG EBOOK.*?\*{3}',
        re.IGNORECASE | re.DOTALL
    )

    # 3. 读取本地文件（适配古腾堡常见编码：utf-8 / latin-1）
    try:
        print(f"正在读取本地文件：{file_path}")
        
        # 优先尝试utf-8编码（古腾堡现代文本常用）
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            print("文件编码：utf-8（读取成功）")
        
        # 若utf-8解码失败，尝试latin-1（古腾堡早期英文文本常用）
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                text = f.read()
            print("文件编码：latin-1（读取成功）")

    # 捕获本地文件读取的常见错误
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 不存在，请检查路径是否正确")
        return None
    except PermissionError:
        print(f"错误：无权限读取文件 {file_path}，请检查文件权限")
        return None
    except Exception as e:
        print(f"错误：读取文件时发生未知错误 - {str(e)[:50]}...")
        return None

    # 4. 提取正文（基于古腾堡标记）
    start_match = start_pat.search(text)
    end_match = end_pat.search(text)

    # 处理标记不存在/位置异常的情况
    if not start_match:
        print("警告：未找到古腾堡正文开始标记（*** START OF ... ***）")
        # 可选：返回全文（若用户希望保留完整文件内容）
        # return text.strip()
        return None
    if not end_match:
        print("警告：未找到古腾堡正文结束标记（*** END OF ... ***）")
        # 可选：返回开始标记后的内容
        # return text[start_match.end():].strip()
        return None
    if start_match.end() >= end_match.start():
        print("错误：开始标记位置在结束标记之后，无法提取正文")
        return None

    # 提取标记间的正文（去除首尾空格）
    content = text[start_match.end():end_match.start()].strip()
    print(f"正文提取成功！正文长度：{len(content)} 字符")
    return content

from pathlib import Path  # 确保已导入

def save_processed_text(original_path, processed_text):
    """
    将预处理后的文本保存至新文件，命名规则：原文件xxx.txt → xxx-p.txt
    
    参数：
        original_path (str): 原文件路径（如 './data/1342-0.txt'）
        processed_text (str): 预处理后的文本内容
    """
    # 转换为Path对象，方便处理路径
    orig_path = Path(original_path)
    
    # 生成新文件名：原文件名（不含扩展名）+ "-p" + 扩展名
    new_filename = f"{orig_path.stem}-p{orig_path.suffix}"  # 如 "1342-0-p.txt"
    
    # 构建新文件的完整路径（与原文件同目录）
    new_path = orig_path.parent / new_filename  # 如 './data/1342-0-p.txt'
    
    try:
        # 确保目标目录存在（若不存在则创建）
        new_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入预处理后的文本（使用utf-8编码，避免中文乱码）
        with open(new_path, 'w', encoding='utf-8') as f:
            f.write(processed_text)
        
        print(f"✅ 预处理文本已保存至：{new_path}")
    except Exception as e:
        print(f"❌ 保存文件失败 {new_path}：{str(e)}")

def test_english_contractions(processed: str)->list:
    """
    检查处理后的文本中是否仍包含未扩展的英文缩写格式
    
    参数：
        processed: 处理后的文本
    返回：
        错误列表（包含仍存在的缩写格式）
    """
    errors = []
    # 统一转为小写检查（忽略大小写影响）
    processed_lower = processed.lower()
    
    # 遍历contractions包支持的所有缩写
    for contraction in contractions.contractions_dict.keys():
        # 1. 检查处理后的文本中是否存在该缩写（完整单词匹配）
        # 正则： 确保匹配独立单词，re.escape处理特殊字符（如撇号'）
        contraction_pattern = re.compile(rf'{re.escape(contraction)}')
        if contraction_pattern.search(processed_lower):
            # 若存在未扩展的缩写，记录错误
            errors.append(f"处理后的文本仍包含未扩展的缩写: '{contraction}'")
    
    return errors

def test_stopwords(processed:str, lang='en')->list:
    """
    假设processed是已经切词用空格分隔连接的字符串
    """
    errors = []

    if lang == 'en':
        stopwords_en = set(nltk.corpus.stopwords.words('english'))
    elif lang == 'zh':
        stopwords_zh = set(nltk.corpus.stopwords.words('chinese'))
    else: 
        raise ValueError("Unsupported language. Use 'en' or 'zh'.")
            
    # tokens = nltk.word_tokenize(processed) if lang == 'en' else jieba.lcut(processed)
    tokens = processed.split()

    if lang == 'en' and any(token in stopwords_en for token in tokens):
        errors.append("英文停用词未移除")
    if lang == 'zh' and any(token in stopwords_zh for token in tokens):
        errors.append("中文停用词未移除")
        
    return errors

def test_preprocessed_text(processed, lang)->list:
    """验证预处理结果"""
    errors = []
    # 检查HTML标签是否移除（假设原始文本含HTML标签，此处简化为检查特殊标签字符）
    if '<' in processed or '>' in processed:
        errors.append("HTML标签未完全移除")
    
    # 检查小写转换
    if any(c.isupper() for c in processed) and lang == 'en':
        errors.append("英文文本未转为小写")
    
    # 检查英文缩写扩展（示例：don't → do not）
    if lang == 'en':
        err_test_english_contractions = test_english_contractions(processed) 
        if len(err_test_english_contractions)>0: # not empty
            errors.extend(err_test_english_contractions)
    
    # 检查停用词移除（英文示例：'the'；中文示例：'的'）
    err_test_stopwords = test_stopwords(processed, lang)
    if len(err_test_stopwords)>0:
        errors.extend(err_test_stopwords)

    # 检查特殊字符移除（示例：'!' '?' 等）
    if any(c in '!@#$%^&*()_+' for c in processed):
        errors.append("特殊字符未移除")
    
    return errors

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="处理书籍列表并生成结果（配置文件为JSON格式）")
    parser.add_argument('config_file', help="书籍列表配置文件路径（如 booklist.json）")
    args = parser.parse_args()

    # 读取JSON配置
    try:
        with open(args.config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
    except Exception as e:
        print(f"读取JSON文件失败: {str(e)}")
        return
    
    # 解析配置数据
    book_dict = dict(config.get('booklist', []))
    params_dict = dict(config.get('preprocessing_params', []))
    
    print("# 预处理结果报告") 
    
    for book_name, file_path in tqdm(book_dict.items(), desc="处理书籍"):
        print(f"\n## 处理书籍：《{book_name}》")
        print(f"\n### 书籍路径和处理参数")
        print(f"* 源文件路径: {file_path}")

        # 获取预处理参数
        if book_name not in params_dict:
            print(f"警告: 未找到 {book_name} 的预处理参数，使用默认值")
            params = {
                "html_stripping": True,
                "contraction_expansion": True,
                "accented_char_removal": True,
                "text_lower_case": True,
                "text_lemmatization": True,
                "special_char_removal": True,
                "stopword_removal": True,
                "remove_digits": False,
                "zh_simplification": True,
                "isDebug": False
            }
        else:
            params = params_dict[book_name]
            print("* 预处理参数:")
            for key, value in params.items():
                print(f"  * {key}: {value}")
            print()        

        # 1. 读取数据内容
        original_doc = fetch_gutenberg_text(file_path)
        if not original_doc:
            print(f"错误： 无法读取书籍内容，跳过处理")
            continue    
                
        print("###1. 成功获取文本")
        print(f"* 长度{len(original_doc)}字符")
        
        # 2. 执行预处理
        processed_doc, lang = normalize_doc(
            original_doc, **params     # **params 将字典解包为关键字参数         
        )

        print("###2. 预处理完成")
        print(f"* 语言：{lang}，处理后长度：{len(processed_doc)}字符")
           
        if lang == 'unknown' or not processed_doc:
            print("错误: 跳过检查（未知语言或空文本）")
            continue

        # 3. 结果输出存储
        print("###3. 结果输出存储")
        save_processed_text(file_path, processed_doc)  # 调用保存函数

        # 4. 输出前10高频词及前20长的单词
        n, k = 10, 20
        top_n, longest_k = get_statistics(processed_doc, n=n, k=k, lang=lang)
        print(f"###4. 输出文本前{n}高频词和前{k}长的单词:")
        print(f"* 前{n}高频词统计")
        print("| 词语 | 出现频率 |")
        print("|------|----------|")
        for word, freq in top_n:
            print(f"| {word} | {freq} |")
        print()
        
        print(f"* 前{k}长词统计")
        print("| 词语 | 长度 |")
        print("|------|----------|")
        for word, len_w in longest_k:
            print(f"| {word} | {len_w} |")
        print()

        print(f"## 检查预处理结果：《{book_name}》")            
        
        # 获取原始文本和处理后文本
        errors = test_preprocessed_text(processed_doc, lang) if processed_doc else []
        if not errors:
            print("✅ 所有检查通过")
        else:
            print("❌ 错误：")
            for e in errors:
                print(f"- {e}")

if __name__ == "__main__":
    # 执行主函数
    main()