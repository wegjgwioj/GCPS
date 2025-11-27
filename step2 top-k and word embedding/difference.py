import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ================= 1. 文件路径配置 =================
# 请替换为你实际生成的两个 Excel 文件路径
# 1. 减数 (BERT / 新方法)
file_bert = r'C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\bert\Task2_BERT_Similarity.xlsx' 
# 2. 被减数 (Keyword / 旧方法)
file_keyword = r'C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\keyword_based_cosine_weighted\Task2_Weighted_Cosine_Similarity.xlsx'

# 检查文件是否存在
if not os.path.exists(file_bert) or not os.path.exists(file_keyword):
    print("错误：找不到输入文件，请检查路径。")
    print(f"BERT路径: {file_bert}")
    print(f"Keyword路径: {file_keyword}")
    exit()

# ================= 2. 读取与对齐数据 =================
print("正在读取数据...")
df_bert = pd.read_excel(file_bert, index_col=0)
df_kw = pd.read_excel(file_keyword, index_col=0)

# 确保两个矩阵的国家顺序完全一致
# 取交集
common_countries = df_bert.index.intersection(df_kw.index)
# 重新排序和筛选
df_bert = df_bert.loc[common_countries, common_countries]
df_kw = df_kw.loc[common_countries, common_countries]

print(f"共对齐 {len(common_countries)} 个国家的数据。")

# ================= 3. 计算差异矩阵 =================
# 公式：Diff = BERT - Keyword
# 正值(红)：BERT认为更相似 (语义关联 > 字面关联)
# 负值(蓝)：Keyword认为更相似 (字面关联 > 语义关联)
df_diff = df_bert - df_kw

# 保存差异矩阵数据
df_diff.to_excel('Task3_Difference_Matrix.xlsx')
print("差异矩阵已保存为: Task3_Difference_Matrix.xlsx")

# ================= 4. 绘制“差值”热力图 =================
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']  # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示

# 设定颜色映射
# cmap='RdBu_r': Red-Blue-Reversed。
# 红色 = 正值 (High)，蓝色 = 负值 (Low)，白色 = 0
# center=0: 强制让白色对应 0 值，确保红蓝分界准确
# vmin/vmax: 设定范围，建议设为对称，比如 -0.3 到 0.3，这样颜色更鲜艳
max_val = max(abs(df_diff.min().min()), abs(df_diff.max().max()))
bound = max_val * 1.0 # 稍微留点余地，或者直接手动指定如 0.4

sns.heatmap(df_diff, 
            annot=True,       # 显示数值
            fmt='.2f',        # 保留两位小数
            cmap='RdBu_r',    # 红蓝配色 (红正蓝负)
            center=0,         # 关键参数：以0为中心
            vmin=-bound,      # 最小值 (蓝色极值)
            vmax=bound,       # 最大值 (红色极值)
            square=True,      # 强制正方形格子
            linewidths=0.5)   # 格子间距

plt.title('语义增益热力图 (BERT相似度 - 关键词相似度)\n红色=语义关联更强 | 蓝色=字面重合更多', fontsize=15)
plt.tight_layout()

output_img = 'Task3_Difference_Heatmap.png'
plt.savefig(output_img, dpi=300)
print(f"图表已保存为: {output_img}")
plt.savefig(r"C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\Task3_Difference_Heatmap.png", dpi=300)