import pandas as pd

# 读取CSV文件
df = pd.read_csv('./data/videos.csv')

# 只取最前面的video_name一列
new_df = df[['video_name']].copy()

# 在第二列全填0
new_df['new_column_2'] = 0

# 在第三列全填-1
new_df['new_column_3'] = -1

# 打印结果
# print(new_df)

# 如果需要保存结果到新的CSV文件
new_df.to_csv('./data/hybrid_train.csv', index=False)