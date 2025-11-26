import pandas as pd
import os
import numpy as np
from collections import Counter
from wordcloud import WordCloud
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

# ================= 配置区域 =================

INPUT_DIR = r"step2 top-k and word embedding/TOP-K keyword"
OUTPUT_DIR = r"step2 top-k and word embedding/Result_Visualization"
FONT_PATH = r"C:\Windows\Fonts\simhei.ttf" 

# 【配置 1】优先录取的 DF 阈值
PRIORITY_MIN_DF = 3

# 【配置 2】Zipf 比例：以总词汇量的多少作为目标数量 (15%)
ZIPF_RATIO = 0.15

# 画布清晰度
SCALE = 4 

# ===========================================

def create_ellipse_mask(width=1600, height=1200):
    """生成椭圆蒙版"""
    mask = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(mask)
    margin = 20
    draw.ellipse((margin, margin, width-margin, height-margin), fill="black")
    return np.array(mask)

def generate_priority_filled_wordcloud():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if not os.path.exists(FONT_PATH):
        print(f"[错误] 字体路径不对: {FONT_PATH}")
        return

    csv_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('_keywords.csv')]
    if not csv_files:
        print("未找到数据文件。")
        return

    print("正在生成椭圆蒙版...")
    ellipse_mask = create_ellipse_mask(width=1600, height=1000)

    for csv_file in csv_files:
        country_name = csv_file.replace('_keywords.csv', '')
        file_path = os.path.join(INPUT_DIR, csv_file)
        
        print(f"\n正在处理 [{country_name}] ...")

        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            df = pd.read_csv(file_path, encoding='gb18030')

        # 1. 收集所有词 (计算文档频率)
        all_keywords = []
        for kw_str in df['keywords']:
            if pd.isna(kw_str) or kw_str == "": continue
            # 确保单文件内去重
            words = list(set([w.strip() for w in str(kw_str).split(',') if w.strip()]))
            all_keywords.extend(words)

        word_counts = Counter(all_keywords)
        
        # 获取所有词的列表 [('词A', 10), ('词B', 5)...] 按频率降序
        all_items_sorted = word_counts.most_common()
        total_vocab_size = len(all_items_sorted)
        
        if total_vocab_size == 0:
            print(f"  -> {country_name} 数据为空，跳过。")
            continue

        # --- 【核心逻辑修改：双梯队填充策略】 ---

        # 1. 计算 Zipf 目标数量
        target_n = int(total_vocab_size * ZIPF_RATIO)
        
        # 兜底：至少展示 30 个词（防止小国词汇量太少画不出来），且不能超过总数
        min_display = 30
        target_n = max(min_display, target_n)
        target_n = min(target_n, total_vocab_size) # 不能超过实际总数

        # 2. 划分梯队
        # 第一梯队：满足 DF 阈值 (高质量)
        high_priority = [item for item in all_items_sorted if item[1] >= PRIORITY_MIN_DF]
        # 第二梯队：不满足 DF 阈值 (用于凑数)
        low_priority = [item for item in all_items_sorted if item[1] < PRIORITY_MIN_DF]

        final_items = []
        
        # 3. 填充逻辑
        if len(high_priority) >= target_n:
            # 情况 A: 高质量词足够多，只取高质量的前 N 个
            final_items = high_priority[:target_n]
            print(f"  -> [质量极佳] 目标 {target_n} 个，全部来自高频词 (DF>={PRIORITY_MIN_DF})")
        else:
            # 情况 B: 高质量词不够，先拿光所有高质量词，再用低频词补齐
            needed_more = target_n - len(high_priority)
            final_items = high_priority + low_priority[:needed_more]
            print(f"  -> [混合填充] 目标 {target_n} 个 = {len(high_priority)} 个高频词 + {needed_more} 个低频词补位")

        # --- 【逻辑结束】 ---

        # 转为字典供词云使用
        word_freq_dict = dict(final_items)

        # ================= 保存权重数据文件 =================
        try:
            weight_df = pd.DataFrame(final_items, columns=['Keyword', 'Weight'])
            
            # 增加一列标记，方便你查看哪些是补位的
            weight_df['Type'] = weight_df['Weight'].apply(lambda x: 'High_DF' if x >= PRIORITY_MIN_DF else 'Low_DF_Fill')

            csv_save_name = f"{country_name}_Weights.csv"
            csv_save_path = os.path.join(OUTPUT_DIR, csv_save_name)
            
            weight_df.to_csv(csv_save_path, index=False, encoding='utf-8-sig')
            print(f"  -> [数据] 已保存 {target_n} 个词: {csv_save_name}")
            
        except Exception as e:
            print(f"  [警告] 权重文件保存失败: {e}")

        # ================= 生成词云 =================
        wc = WordCloud(
            font_path=FONT_PATH,
            background_color='white',
            mask=ellipse_mask,
            max_words=target_n, # 强制使用计算出的数量
            max_font_size=250,
            min_font_size=10,
            random_state=42,
            prefer_horizontal=0.9,
            colormap='Dark2',
            contour_width=2,
            contour_color='steelblue',
            scale=SCALE
        )

        try:
            wc.generate_from_frequencies(word_freq_dict)
            img_save_name = f"{country_name}_Cloud.png"
            img_save_path = os.path.join(OUTPUT_DIR, img_save_name)
            wc.to_file(img_save_path)
            print(f"  -> [图片] 词云已保存: {img_save_name}")

        except Exception as e:
            print(f"  [失败] {e}")

    print("-" * 30)
    print(f"任务完成，结果路径: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_priority_filled_wordcloud()