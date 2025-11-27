import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

# ================= 1. 数据加载 =================
# 请替换为你实际的文件路径
# file_left = "Left_Image_BERT.xlsx" #(左图：变化后/BERT)
# file_right = "Right_Image_Keyword.xlsx"# (右图：变化前/Keyword)

file_path_after = r"C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\bert\Task2_BERT_Similarity.xlsx"
# 假设右图（旧方法）也保存为了Excel，如果只有代码生成的变量，直接使用变量即可
file_path_before = r"C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\keyword_based_cosine_weighted\Task2_Weighted_Cosine_Similarity.xlsx" 

data_after = pd.read_excel(file_path_after, index_col=0)  # 左图（BERT）
data_before = pd.read_excel(file_path_before, index_col=0) # 右图（Keyword）

# ================= 2. 数据预处理 =================
# 确保两个矩阵的行列顺序一致
common_countries = data_after.index.intersection(data_before.index)
data_after = data_after.loc[common_countries, common_countries]
data_before = data_before.loc[common_countries, common_countries]

def flatten_matrix(df, label_name):
    """
    提取矩阵上三角数据（去除对角线和重复项），转换为列表
    """
    # 获取上三角掩码（不含对角线 k=1）
    mask = np.triu(np.ones(df.shape), k=1).astype(bool)
    # 提取数据并堆叠
    flattened = df.where(mask).stack().reset_index()
    flattened.columns = ['Country_A', 'Country_B', label_name]
    # 创建唯一标识符
    flattened['Pair'] = flattened['Country_A'] + " - " + flattened['Country_B']
    return flattened[['Pair', label_name]]

# 展平两个矩阵
df_flat_after = flatten_matrix(data_after, 'Score_After')
df_flat_before = flatten_matrix(data_before, 'Score_Before')

# 合并数据
df_trend = pd.merge(df_flat_before, df_flat_after, on='Pair')

# ================= 3. 统计分析 =================
# 1. 皮尔逊相关系数 (线性关系)
p_corr, _ = pearsonr(df_trend['Score_Before'], df_trend['Score_After'])

# 2. 斯皮尔曼等级相关系数 (排名的单调性：原本高的现在是否还排在前面？)
s_corr, _ = spearmanr(df_trend['Score_Before'], df_trend['Score_After'])

# 3. 计算排名的变化
df_trend['Rank_Before'] = df_trend['Score_Before'].rank(ascending=False)
df_trend['Rank_After'] = df_trend['Score_After'].rank(ascending=False)
df_trend['Rank_Diff'] = df_trend['Rank_After'] - df_trend['Rank_Before'] # 正数表示排名下降，负数表示排名上升

print("="*30)
print(f"统计分析结果：")
print(f"1. 皮尔逊相关系数 (Pearson): {p_corr:.3f}")
print(f"   (接近1表示数值变化趋势高度一致)")
print(f"2. 斯皮尔曼等级相关 (Spearman): {s_corr:.3f}")
print(f"   (接近1表示原本相关性高的国家对，在新方法中依然排名靠前)")
print("="*30)

# ================= 4. 可视化：趋势散点图 =================
plt.figure(figsize=(10, 8))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False

# 绘制散点图和回归线
sns.regplot(data=df_trend, x='Score_Before', y='Score_After', 
            color='b', scatter_kws={'alpha':0.6}, line_kws={'color':'red'})

# 标注原本相关性最高的 Top 5
top_5_before = df_trend.nlargest(5, 'Score_Before')
for i in range(len(top_5_before)):
    row = top_5_before.iloc[i]
    plt.text(row['Score_Before']+0.01, row['Score_After'], 
             row['Pair'], fontsize=9, color='darkred', weight='bold')

plt.title(f'国家政策相似度计算方法对比\nSpearman Rank Correlation: {s_corr:.2f}', fontsize=14)
plt.xlabel('变化前：基于 Keyword Weight 的相似度 (右图)', fontsize=12)
plt.ylabel('变化后：基于 BERT + Top-K 的相似度 (左图)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# 保存图片
plt.savefig('Correlation_Trend_Analysis.png')
plt.show()

# ================= 5. 输出原本高相关的Top 10在新方法中的表现 =================
print("\n【原本(右图)相似度最高的 Top 10 国家对，在现有(左图)中的表现】：")
top_10 = df_trend.sort_values('Score_Before', ascending=False).head(10)
print(top_10[['Pair', 'Score_Before', 'Rank_Before', 'Score_After', 'Rank_After']].to_string(index=False))

# 检查是否仍然属于“高相关” (例如：新排名前15名内)
print("\n分析结论：")
stable_pairs = top_10[top_10['Rank_After'] <= 15]
if len(stable_pairs) >= 7:
    print(f"趋势保持良好！原本Top10中有 {len(stable_pairs)} 对在BERT方法中依然排名前15。")
else:
    print(f"趋势发生显著变化。原本Top10中仅有 {len(stable_pairs)} 对在BERT方法中依然排名前15。BERT 模型可能捕捉到了关键词之外的深层语义。")