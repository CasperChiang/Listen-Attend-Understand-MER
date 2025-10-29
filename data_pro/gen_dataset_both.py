import os
import pickle
import numpy as np

import pandas as pd



data_path = "../data"

with open('mel_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    for k, v in loaded_data['mel'].items():
        os.makedirs(os.path.join(data_path, k), exist_ok=True) # 创建该条数据的目录

with open('mel_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    for k, v in loaded_data['mel'].items():
        np.save(os.path.join(data_path, k) + '/mel_spectrogram', v)
        print("成功保存", k,  "梅尔图")
        print("\n\n")

with open('coc_data.pkl', 'rb') as file:
    loaded_data = pickle.load(file)
    for k, v in loaded_data['coc'].items():
        np.save(os.path.join(data_path, k) + '/cochleagram', v)
        print("成功保存", k,  "耳蜗图")
        print("\n\n")


valence_file = './valence.csv'
arousal_file = './arousal.csv'

df_valence = pd.read_csv(valence_file)
df_arousal = pd.read_csv(arousal_file)

print("成功加载 CSV 文件，开始处理...")

# 循环处理每一行数据
for i in range(0, 1801):
    # 确保 file_index 是整数类型
    file_index = int(df_valence.iloc[i, 0])

    # 提取 valence 和 arousal 数据，并确保它们是浮点数类型以便计算
    valence = df_valence.iloc[i, 1:61].values.astype(float)
    arousal = df_arousal.iloc[i, 1:61].values.astype(float)

    va = np.column_stack((valence, arousal))

    # 确保保存文件的子目录存在
    save_dir = os.path.join(data_path, str(file_index))
    os.makedirs(save_dir, exist_ok=True)

    # 保存为 .npy 文件
    save_path = os.path.join(save_dir, 'va_sequence.npy')
    np.save(save_path, va)

    print(f"保存了索引为 {file_index} 的 VA 序列到 {save_path}")

print("所有文件处理完成！")


# 数据目录
data_dir = data_path
output_csv = '../data/data.csv'

# 初始化列表存储元数据
data = {
    'audio_id': [],
    'cochleagram_path': [],
    'mel_spectrogram_path': [],
    'va_sequence_path': [],
}

# 遍历数据目录
for audio_folder in sorted(os.listdir(data_dir), key=lambda x: int(x)):
    audio_path = os.path.join(data_dir, audio_folder)
    if os.path.isdir(audio_path):
        # 确保文件夹包含所需文件
        cochleagram_file = os.path.join(audio_path, 'cochleagram.npy')
        mel_spectrogram_file = os.path.join(audio_path, 'mel_spectrogram.npy')
        va_sequence_file = os.path.join(audio_path, 'va_sequence.npy')

        # 检查文件是否存在
        if (os.path.exists(cochleagram_file) and
            os.path.exists(mel_spectrogram_file) and
            os.path.exists(va_sequence_file)):
            data['audio_id'].append(audio_folder)
            data['cochleagram_path'].append(cochleagram_file)
            data['mel_spectrogram_path'].append(mel_spectrogram_file)
            data['va_sequence_path'].append(va_sequence_file)
# 创建 DataFrame
df = pd.DataFrame(data)

# 保存到 CSV
df.to_csv(output_csv, index=False)
print(f"CSV 文件已生成：{output_csv}")
