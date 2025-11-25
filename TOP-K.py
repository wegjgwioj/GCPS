import os
import pandas as pd
import jieba
import jieba.analyse

# ================= 路径配置区域 =================
SOURCE_DIR = r"country"
OUTPUT_DIR = r"TOP-K keyword"
STOPWORDS_PATH = os.path.join(SOURCE_DIR, "cn_stopwords.txt")

# ================= 提取策略配置 =================
ALLOWED_POS = ('vn', 'n', 'nr', 'ns', 'nt','vn', 'nz')

# 【核心新增 1】固定词表 (强制不分词)
# 在这里填入你不想被 jieba 切开的词，比如人名、特定政策、长专有名词
FIXED_WORDS = [
    "一带一路", "碳中和", "供应链", "人工智能", "二十国集团", 
    "可再生能源", "泽连斯基", "默克尔",
    "社会民主党", "自由民主党", "绿色经济", "气候变化",
    "命运共同体", "大流行", "通货膨胀","温室气体","巴黎协定","巴黎大会"
    "贸易战","贸易制裁","贸易保护主义","绿水青山就是金山银山","绿水青山"
]

# 国家全称 -> 简称映射表
COUNTRY_SHORT_MAP = {
    "德国": "德",
    "意大利": "意",
    "日本": "日",
    "韩国": "韩",
    "沙特阿拉伯": "沙特",
    "印度尼西亚": "印尼",
    "美国": "美",
    "英国": "英",
    "法国": "法",
    "中国": "中"
}

# ==============================================

def get_base_stopwords():
    """读取基础停用词表"""
    stopwords_set = set()
    if os.path.exists(STOPWORDS_PATH):
        try:
            with open(STOPWORDS_PATH, 'r', encoding='utf-8') as f:
                stopwords_set = set(line.strip() for line in f)
        except Exception:
            with open(STOPWORDS_PATH, 'r', encoding='gb18030') as f:
                stopwords_set = set(line.strip() for line in f)
    return stopwords_set

def init_jieba_environment():
    """
    【核心新增 2】初始化 Jieba 环境
    加载固定词表，确保这些词不会被切分
    """
    print("正在初始化 Jieba 词典...")
    count = 0
    for word in FIXED_WORDS:
        # add_word 强制让 jieba 记住这个词是一个整体
        jieba.add_word(word)
        count += 1
    print(f"-> 已加载 {count} 个固定词汇 (如: {FIXED_WORDS[0]}...)")


def extract_and_save_to_target():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. 初始化环境 (加载固定词表)
    init_jieba_environment()

    # 2. 读取基础停用词
    base_stopwords = get_base_stopwords()
    temp_stopwords_file = os.path.join(OUTPUT_DIR, "temp_dynamic_stopwords.txt")

    for entry in os.listdir(SOURCE_DIR):
        country_dir = os.path.join(SOURCE_DIR, entry)
        if not os.path.isdir(country_dir):
            continue

        print(f"\n正在处理国家: {entry} ...")
        
        short_name = COUNTRY_SHORT_MAP.get(entry, entry[0])
        
        # 动态停用词
        current_dynamic_stops = {
            f"{entry}政府",
            f"{entry}官员",
            f"{entry}企业",
            short_name,
            f"{short_name}方",
            f"{short_name}媒",
            f"{short_name}国",
            f"{short_name}政府",
        }
        
        combined_stopwords = base_stopwords.union(current_dynamic_stops)

        try:
            with open(temp_stopwords_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(combined_stopwords))
            jieba.analyse.set_stop_words(temp_stopwords_file)
            print(f"  -> 已应用动态停用词 (含简称 '{short_name}' 系列)")
        except Exception as e:
            print(f"  [警告] 动态停用词应用失败: {e}")

        # -------------------- 提取逻辑 --------------------
        results = []
        file_list = [f for f in os.listdir(country_dir) if f.lower().endswith('.txt')]

        if not file_list:
            continue

        for file_name in file_list:
            file_path = os.path.join(country_dir, file_name)
            try:
                content = ""
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
                except UnicodeDecodeError:
                    with open(file_path, 'r', encoding='gb18030') as f: content = f.read()
                
                content_clean = content.replace('\n', '').strip()
                text_len = len(content_clean)
                if not content_clean: continue

                # 动态 TopK
                if text_len < 100: target_top_k = 3
                elif text_len < 300: target_top_k = 5
                else: target_top_k = 10

                # 1. 提取两倍候选词
                candidate_k = target_top_k * 2

                # 2. 算法A: TF-IDF
                kw_tfidf = jieba.analyse.extract_tags(
                    content_clean, topK=candidate_k, withWeight=False, allowPOS=ALLOWED_POS
                )
                
                # 3. 算法B: TextRank
                kw_textrank = jieba.analyse.textrank(
                    content_clean, topK=candidate_k, withWeight=False, allowPOS=ALLOWED_POS
                )

                # 4. 取交集 (Intersection)
                intersection = [w for w in kw_textrank if w in kw_tfidf]
                
                # 5. 补充词
                only_textrank = [w for w in kw_textrank if w not in intersection]
                only_tfidf = [w for w in kw_tfidf if w not in intersection]
                
                # 6. 合并
                combined_keywords = intersection + only_textrank + only_tfidf
                
                # 7. 截取
                final_keywords = combined_keywords[:target_top_k]

                results.append({
                    'file_name': file_name,
                    'keywords': ",".join(final_keywords),
                    'count': len(final_keywords),
                    'text_length': text_len
                })
            except Exception:
                pass

        if results:
            df = pd.DataFrame(results)
            save_path = os.path.join(OUTPUT_DIR, f"{entry}_keywords.csv")
            df.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"  -> [完成] 已保存至: {save_path}")

    if os.path.exists(temp_stopwords_file):
        os.remove(temp_stopwords_file)

if __name__ == "__main__":
    extract_and_save_to_target()