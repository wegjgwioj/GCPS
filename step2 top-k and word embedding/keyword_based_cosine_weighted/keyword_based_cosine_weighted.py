import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# ================= 1. 路径配置 =================
# 根据你的截图，CSV 文件在 'country-keyword' 文件夹下
folder_path = r'./step2 top-k and word embedding/country-keyword' 

# 检查路径
if not os.path.exists(folder_path):
    print(f"错误：找不到路径 {folder_path}，请检查文件夹名称是否完全一致。")
    exit()

# ================= 2. 读取并聚合数据 =================
# 匹配文件名模式，例如 "德国_Weights.csv"
file_pattern = os.path.join(folder_path, '*_Weights.csv')
files = glob.glob(file_pattern)

if not files:
    print("错误：未找到 _Weights.csv 文件！")
    exit()

print(f"检测到 {len(files)} 个国家文件，开始构建加权向量...")

# 我们用一个字典来收集数据： {'德国': {'能源': 77, '气候': 41}, '法国': {...}}
data_dict = {}

for file in files:
    # 提取国家名
    file_name = os.path.basename(file)
    country_name = file_name.split('_')[0]
    
    try:
        # 读取 CSV
        df = pd.read_csv(file)
        
        # 确保列名没有空格
        df.columns = [c.strip() for c in df.columns]
        
        # 检查必要的列是否存在 (根据你的截图，列名是 'Keyword' 和 'Weight')
        if 'Keyword' not in df.columns or 'Weight' not in df.columns:
            print(f"警告：{file_name} 缺少 'Keyword' 或 'Weight' 列，跳过。")
            continue
            
        # 将该国数据转为字典格式 {关键词: 权重}
        # set_index('Keyword')['Weight'] 把关键词设为索引，取权重列
        country_weights = df.set_index('Keyword')['Weight'].to_dict()
        
        data_dict[country_name] = country_weights
        print(f"  -> {country_name}: 加载成功 ({len(country_weights)} 个词)")
        
    except Exception as e:
        print(f"  -> 读取 {file_name} 失败: {e}")

# ================= 3. 构建矩阵 & 计算余弦相似度 =================
print("\n正在对齐数据并计算相似度...")

# 将字典转为 DataFrame (自动对齐所有关键词)
# 行=国家，列=关键词，值=权重
# fillna(0) 很重要：如果德国没提"埃菲尔铁塔"，权重补0
df_matrix = pd.DataFrame(data_dict).T.fillna(0)

# 计算余弦相似度
similarity_matrix = cosine_similarity(df_matrix)

# ================= 4. 输出结果 =================
# 4.1 保存 Excel
df_sim = pd.DataFrame(similarity_matrix, index=df_matrix.index, columns=df_matrix.index)
output_file = 'Weighted_Cosine_Similarity.xlsx'
df_sim.to_excel(output_file)
print(f"\n[成功] 相似度矩阵已保存: {output_file}")

# 4.2 画热力图 (Task 3 预览)
try:
    plt.figure(figsize=(12, 10))
    # 设置中文支持
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制热力图
    # cmap='RdYlBu_r' 红蓝配色，红色代表高相似，蓝色代表低相似
    sns.heatmap(df_sim, annot=True, cmap='RdYlBu_r', fmt='.2f', vmin=0, vmax=1)
    
    plt.title('国家政策加权相似度 (基于Keyword Weight)', fontsize=16)
    plt.tight_layout()
    plt.savefig('Task3_Weighted_Heatmap.png')
    print(f"[成功] 热力图已保存: Task3_Weighted_Heatmap.png")
    # plt.show() 
except Exception as e:
    print(f"绘图报错: {e}")

print("\n任务完成！")