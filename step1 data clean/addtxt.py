import pandas as pd
import os

def process_csv(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 {file_path}")
        return

    try:
        # 1. 读取 CSV 文件
        # 注意：由于你的示例中包含中文和乱码，这里尝试用 'utf-8' 读取，
        # 如果报错则尝试 'gb18030' (兼容GBK)，以防止编码错误。
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except UnicodeDecodeError:
            print("UTF-8读取失败，尝试使用 GB18030 编码读取...")
            df = pd.read_csv(file_path, encoding='gb18030')

        print("文件读取成功，正在处理...")

        # 2. 将原有的 'fileId' 列重命名为 'orgin_fileId'
        if 'fileId' in df.columns:
            df.rename(columns={'fileId': 'orgin_fileId'}, inplace=True)
        else:
            print("警告：文件中未找到 'fileId' 列，请检查表头。")
            # 如果没有这一列，后续操作可能会失败，这里做个简单的容错处理
            if 'orgin_fileId' not in df.columns:
                return

        # 3. 生成新的 fileId 内容
        # 逻辑：text_ + orgin_fileId + .txt
        new_file_id_values = 'text_' + df['orgin_fileId'].astype(str) + '.txt'

        # 4. 在第一列插入新的 'fileId' 列
        # insert(插入位置索引, 列名, 列内容)
        df.insert(0, 'fileId', new_file_id_values)

        # 5. 保存文件
        # 构造输出路径，避免直接覆盖原文件
        dir_name = os.path.dirname(file_path)
        file_name = os.path.basename(file_path)
        output_path = os.path.join(dir_name, file_name.replace('.csv', '_processed.csv'))

        # index=False 表示不保存行号
        # encoding='utf-8-sig' 可以解决 Excel 打开中文乱码的问题
        df.to_csv(output_path, index=False, encoding='utf-8-sig')

        print(f"处理完成！")
        print(f"新文件已保存至: {output_path}")
        print("-" * 30)
        print("前5行预览：")
        print(df[['fileId', 'orgin_fileId']].head())

    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 设置文件路径
    # 注意：Windows路径建议在引号前加 r，或者将 \ 改为 \\
    target_file = r"GCPS/step1 data clean/orign data/英国/英国.csv"
    
    process_csv(target_file)