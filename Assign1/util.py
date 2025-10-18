# -*- coding: utf-8 -*-
# @Author: Haiqin Yang
# @Date:   2025-10-07 16:51:32
# @Last Modified by:   Haiqin Yang
# @Last Modified time: 2025-10-18 19:32:15
import nltk
import jieba
import contractions
import re
from pathlib import Path  # 确保已导入

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
    fdist = nltk.FreqDist(tokens)
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