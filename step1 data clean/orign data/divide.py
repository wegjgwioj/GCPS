import pandas as pd
import os

# ================= 配置区域 =================

# 注意：如果你是在终端直接运行，请确保下面的路径和文件名是正确的
# 建议直接修改为你报错截图中的那个具体文件路径，或者保持现状由脚本自动寻找
INPUT_FILE = r'C:\Users\Andy\Desktop\期中作业双碳数据集\德意\2德意.csv' 
OUTPUT_FILE = r'C:\Users\Andy\Desktop\期中作业双碳数据集\德意\德意.csv'  # 结果文件名

# 关键词映射配置
KEYWORDS_MAP = {
    "意大利": "意大利",
    "Italy": "意大利",
    "意大": "意大利",
    "法意": "意大利",
    "德国": "德国",
    "Germany": "德国",
}

# ================= 代码逻辑 =================

def process_csv(input_path, output_path):
    print(f"正在读取文件: {input_path} ...")
    
    df = None
    
    # 尝试方案 1: 使用 GB18030 (比GBK支持更多字符) 并忽略错误
    try:
        df = pd.read_csv(
            input_path, 
            dtype=str, 
            on_bad_lines='skip', 
            encoding='gb18030', 
            encoding_errors='replace'  # 关键修改：遇到乱码替换成  而不是报错
        )
        print("成功使用 GB18030 编码读取。")
    except Exception as e:
        print(f"GB18030 读取失败，原因: {e}")
    
    # 尝试方案 2: 如果上面失败，尝试 UTF-8 并忽略错误
    if df is None:
        try:
            print("尝试使用 UTF-8 编码读取...")
            df = pd.read_csv(
                input_path, 
                dtype=str, 
                on_bad_lines='skip', 
                encoding='utf-8', 
                encoding_errors='replace' # 关键修改
            )
        except Exception as e:
            print(f"UTF-8 读取失败，原因: {e}")
            return # 彻底失败

    cleaned_data = []

    # 2. 遍历每一行
    total_rows = len(df)
    print(f"文件读取成功，共 {total_rows} 行，开始清洗...")

    for index, row in df.iterrows():
        if index % 1000 == 0:
            print(f"处理进度: {index}/{total_rows}")

        # 获取 fileId (尝试获取第1列，如果为空则跳过)
        try:
            # 确保取到的是字符串
            file_id_raw = row.iloc[0]
            if pd.isna(file_id_raw):
                continue
            file_id = str(file_id_raw).strip()
        except:
            continue

        # 将这一行的所有列内容拼接成一个大字符串进行搜索
        row_content = " ".join(row.fillna('').astype(str).tolist())
        
        matched_country = "未识别"
        
        for keyword, country_name in KEYWORDS_MAP.items():
            if keyword in row_content:
                matched_country = country_name
                break 
        
        cleaned_data.append({
            "fileId": file_id,
            "country": matched_country
        })

    # 3. 生成结果
    result_df = pd.DataFrame(cleaned_data)
    
    # 保存时使用 utf-8-sig 防止Excel乱码
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"处理完成！结果已保存至: {output_path}")
    print(f"共提取数据: {len(result_df)} 条")

if __name__ == "__main__":
    # 检查文件是否存在
    if os.path.exists(INPUT_FILE):
        process_csv(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"错误：找不到文件 -> {INPUT_FILE}")
        print("请在代码的 'INPUT_FILE' 处修改为正确的文件路径。")