import pandas as pd
import os
import shutil

# ================= 核心配置区域 =================

# 1. 基础路径
BASE_DIR = r"GCPS/step1 data clean/orign data"

# 2. 当前任务文件夹
# 选项: "德意", "日韩沙特印尼"
CURRENT_TASK_FOLDER = "日韩沙特印尼"

# 3. 国家关键词规则配置
# 逻辑：脚本会统计关键词出现的频率，将文件归类到频率最高的国家
TASK_CONFIG = {
    # === 德意 组配置 (如需使用请取消注释并修改 CURRENT_TASK_FOLDER) ===
    # "德国": ["德国", "德意志", "柏林", "法兰克福", "Germany", "Deutsch", "默克尔", "朔尔茨", "中德", "德媒"],
    # "意大利": ["意大利", "意国", "罗马", "米兰", "Italy", "Italian", "意大", "中意", "意媒"],
    
    # === 日韩沙特印尼 组配置 ===
    "日本": ["日本", "东京", "大阪", "Japan", "Jp", "日媒", "中日", "安倍", "岸田"],
    "韩国": ["韩国", "首尔", "Korea", "KR", "韩媒", "中韩", "文在寅", "尹锡悦"],
    "沙特": ["沙特", "利雅得", "Saudi", "Riyadh", "沙特阿拉伯", "本·萨勒曼"],
    "印尼": ["印尼", "印度尼西亚", "雅加达", "Indonesia", "佐科"],
}

# 4. 未分类文件夹的名称
UNCLASSIFIED_NAME = "未分类"

# 5. 需要用于搜索关键词的列 (按优先级排序，合并在一起进行统计)
SEARCH_COLUMNS = ['area', 'title', 'keywords', 'description', 'news_category', 'source']

# ==============================================

def process_and_copy_files():
    # 构造工作路径
    work_dir = os.path.join(BASE_DIR, CURRENT_TASK_FOLDER)
    csv_path = os.path.join(work_dir, f"{CURRENT_TASK_FOLDER}.csv")

    # 检查CSV是否存在
    if not os.path.exists(csv_path):
        print(f"[错误] 找不到CSV文件: {csv_path}")
        return

    # 1. 读取 CSV
    print(f"正在读取 CSV: {csv_path} ...")
    try:
        df = pd.read_csv(csv_path, encoding='utf-8')
    except UnicodeDecodeError:
        print("UTF-8读取失败，切换为 GB18030 读取...")
        df = pd.read_csv(csv_path, encoding='gb18030')

    # 2. 初始化数据容器和目录
    all_categories = list(TASK_CONFIG.keys()) + [UNCLASSIFIED_NAME]
    data_buckets = {cat: [] for cat in all_categories}

    # 创建目标文件夹
    print("正在创建目标文件夹...")
    for category in all_categories:
        dir_path = os.path.join(work_dir, category)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    # 3. 遍历并处理
    print("-" * 30)
    print("开始基于关键词频次进行分类...")
    
    stats = {'success_copy': 0, 'missing_file': 0}

    for index, row in df.iterrows():
        file_id = str(row.get('fileId', '')).strip()
        
        # 如果没有 fileId，跳过
        if not file_id or file_id.lower() == 'nan':
            continue

        # --- 步骤 A: 确定分类 (核心修改部分) ---
        
        # 1. 准备文本：将所有搜索列合并为一个长字符串，并转小写
        content_text = " ".join([str(row.get(col, '')) for col in SEARCH_COLUMNS]).lower()
        
        # 2. 计算每个国家的得分
        scores = {}
        for country, keywords in TASK_CONFIG.items():
            count = 0
            for kw in keywords:
                # 统计关键词出现的次数 (例如 'Korea' 出现了 3 次)
                # 注意：kw也要转小写以匹配 content_text
                count += content_text.count(kw.lower())
            scores[country] = count
        
        # 3. 找出最高分
        # max_score 是最高的分数
        # best_country 是最高分对应的国家名
        if not scores:
            best_country = UNCLASSIFIED_NAME
            max_score = 0
        else:
            # key=scores.get 表示按照字典的值来比较
            best_country = max(scores, key=scores.get)
            max_score = scores[best_country]

        # 4. 判定最终归属
        if max_score > 0:
            target_category = best_country
        else:
            target_category = UNCLASSIFIED_NAME
        
        # 将行数据加入对应列表
        data_buckets[target_category].append(row)

        # --- 步骤 B: 复制文件 ---
        src_file_path = os.path.join(work_dir, file_id)
        dst_dir_path = os.path.join(work_dir, target_category)
        dst_file_path = os.path.join(dst_dir_path, file_id)

        if os.path.exists(src_file_path):
            try:
                shutil.copy2(src_file_path, dst_file_path)
                stats['success_copy'] += 1
            except Exception as e:
                print(f"[复制失败] {file_id}: {e}")
        else:
            stats['missing_file'] += 1

    # 4. 生成各分类的 CSV 文件
    print("-" * 30)
    print("正在生成分类 CSV 文件...")

    for category, rows in data_buckets.items():
        if len(rows) == 0:
            continue
            
        sub_df = pd.DataFrame(rows)
        save_path = os.path.join(work_dir, category, f"{category}.csv")
        
        sub_df.to_csv(save_path, index=False, encoding='utf-8-sig')
        print(f"  [{category}] 类: {len(sub_df)} 条数据 (已保存)")

    # 5. 总结
    print("=" * 30)
    print("任务完成报告:")
    print(f"成功复制文件数: {stats['success_copy']}")
    print(f"源文件缺失数  : {stats['missing_file']}")
    print("-" * 15)
    print("分类详情:")
    for cat, rows in data_buckets.items():
        print(f"  - {cat}: {len(rows)}")

if __name__ == "__main__":
    process_and_copy_files()