import os
import glob
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer, util
import seaborn as sns
import matplotlib.pyplot as plt

# ================= 1. 路径配置 =================
folder_path = r'step2 top-k and word embedding/country-keyword'
file_pattern = os.path.join(folder_path, '*_Weights.csv')
files = glob.glob(file_pattern)

if not files:
    raise FileNotFoundError("没有找到 *_Weights.csv 文件，请检查路径")

print(f"检测到 {len(files)} 个国家文件")

# ================= 2. Sentence-BERT 模型 =================
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# ================= 3. 权重系数（High/Low 调整） =================
HIGH_MULTIPLIER = 1.3    # High_DF 的权重增益
LOW_MULTIPLIER  = 0.7    # Low_DF 的权重衰减

# ================= 4. 生成国家文本 =================
country_texts = {}

for file in files:
    df = pd.read_csv(file)
    df.columns = [c.strip() for c in df.columns]  # 清理空格

    name = os.path.basename(file).split('_')[0]  # 国家名

    text_lines = []
    for _, row in df.iterrows():
        kw = str(row["Keyword"]).strip()
        w  = float(row["Weight"])
        tp = str(row["Type"]).strip()

        # High/Low 权重处理
        if "High" in tp:
            w_final = w * HIGH_MULTIPLIER
            tp_desc = "属于高文档频率（由多篇政策文档共同出现）"
        else:
            w_final = w * LOW_MULTIPLIER
            tp_desc = "属于低文档频率（可能只在单篇文档出现）"

        text_lines.append(
            f"关键词“{kw}”，权重{w_final:.2f}，{tp_desc}。"
        )

    # 拼成最终国家文本（让 BERT 读）
    country_texts[name] = "\n".join(text_lines)
    print(f"[完成] {name} 文本生成，共 {len(df)} 条")

# ================= 5. BERT 编码 =================
country_embeddings = {}

print("\n开始生成 BERT 向量...")
for name, text in country_texts.items():
    emb = model.encode(text, convert_to_tensor=True)
    country_embeddings[name] = emb

# ================= 6. 国家相似度矩阵 =================
country_names = list(country_embeddings.keys())
mat = np.zeros((len(country_names), len(country_names)))

for i, c1 in enumerate(country_names):
    for j, c2 in enumerate(country_names):
        mat[i][j] = util.cos_sim(country_embeddings[c1], country_embeddings[c2]).item()

df_sim = pd.DataFrame(mat, index=country_names, columns=country_names)

# 保存 Excel
df_sim.to_excel(r"C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\bert\Task2_BERT_Similarity.xlsx")
print("\n[成功] 相似度矩阵保存为 Task2_BERT_Similarity.xlsx")

# ================= 7. 热力图 =================
plt.figure(figsize=(12, 10))
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# --- 修改重点 ---
# 添加 vmin=0 和 vmax=1 以固定颜色条范围为 0 到 1
sns.heatmap(df_sim, annot=True, cmap='RdYlBu_r', fmt=".2f", vmin=0.5, vmax=1)
# ----------------

plt.title("国家政策语义相似度（BERT + Top-K 关键词 + High/Low 加权）")
plt.tight_layout()
plt.savefig(r"C:\Users\Andy\Desktop\文本分析\GCPS\step2 top-k and word embedding\bert\Task3_BERT_Heatmap  2.png")
print("[成功] 热力图保存为 Task3_BERT_Heatmap.png")

print("\n任务完成！")