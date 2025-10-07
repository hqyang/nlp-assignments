# -*- coding: utf-8 -*-
# @Author: Haiqin Yang
# @Date:   2025-10-05 18:20:05
# @Last Modified by:   Haiqin Yang
# @Last Modified time: 2025-10-07 17:04:14

#!/usr/bin/env python3

from tqdm import tqdm  # 导入tqdm库
import argparse
import json
from pathlib import Path  # 用于路径验证
from Assign1_func import normalize_doc  # 导入预处理函数
from util import * # 导入需要的函数

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