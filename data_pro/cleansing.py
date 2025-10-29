import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # 一个好用的进度条库，如果未安装请运行: pip install tqdm

# --- 1. 配置参数 ---
# 请确保这个路径是正确的
METADATA_PATH = '../data/data.csv'

# 定义您的数据应该有的标准形状
EXPECTED_MEL_SHAPE = (96, 2640)
EXPECTED_COCH_SHAPE = (11, 2640)


def fix_npy_file(file_path, expected_shape):
    """
    加载一个 .npy 文件, 检查其形状。如果时间维度不匹配，则用0填充并覆盖保存。

    参数:
    - file_path (str): .npy 文件的路径。
    - expected_shape (tuple): 期望的文件形状, 如 (96, 2640)。

    返回:
    - bool: 如果文件被修复则返回 True, 否则返回 False。
    """
    if not os.path.exists(file_path):
        print(f"  [警告] 文件不存在，跳过: {file_path}")
        return False

    try:
        data = np.load(file_path)
        current_shape = data.shape

        # 检查形状是否已经符合预期
        if current_shape == expected_shape:
            return False

        # 检查特征维度是否正确
        if current_shape[0] != expected_shape[0]:
            print(f"  [错误] 文件特征维度不匹配，无法修复: {file_path}, 形状为 {current_shape}")
            return False

        # 检查时间维度是否过长或过短
        if current_shape[1] > expected_shape[1]:
            print(f"  [警告] 文件时间维度过长，将被截断: {file_path}, 形状为 {current_shape}")
            fixed_data = data[:, :expected_shape[1]]
        elif current_shape[1] < expected_shape[1]:
            # 计算需要在时间维度末尾填充的长度
            padding_width = expected_shape[1] - current_shape[1]

            # 定义填充方式：只在第二个维度（时间轴）的末尾填充
            # ((0, 0), (0, padding_width)) 表示：
            # - 第一个维度（特征轴）前后都不填充
            # - 第二个维度（时间轴）前面不填充，末尾填充 padding_width 个值
            paddings = ((0, 0), (0, padding_width))

            # 使用 'constant' 模式填充0
            fixed_data = np.pad(data, pad_width=paddings, mode='constant', constant_values=0)
        else:
            # 长度正好，无需操作
            return False

        # 覆盖保存修复后的文件
        np.save(file_path, fixed_data)
        print(f"  [已修复] {os.path.basename(file_path)}: 原始形状 {current_shape} -> 目标形状 {fixed_data.shape}")
        return True

    except Exception as e:
        print(f"  [错误] 处理文件时发生异常 {file_path}: {e}")
        return False


# --- 2. 主执行逻辑 ---
if __name__ == "__main__":
    print(f"开始扫描并修复数据，元数据文件: {METADATA_PATH}")

    try:
        metadata = pd.read_csv(METADATA_PATH)
    except FileNotFoundError:
        print(f"[致命错误] 元数据文件未找到: {METADATA_PATH}")
        exit()

    mel_paths = metadata['mel_spectrogram_path'].values
    coch_paths = metadata['cochleagram_path'].values

    mel_fixed_count = 0
    coch_fixed_count = 0

    print("\n--- 正在检查梅尔图 (Mel Spectrograms) ---")
    for path in tqdm(mel_paths, desc="处理梅尔图"):
        if fix_npy_file(path, EXPECTED_MEL_SHAPE):
            mel_fixed_count += 1

    print(f"\n--- 正在检查耳蜗图 (Cochleagrams) ---")
    for path in tqdm(coch_paths, desc="处理耳蜗图"):
        if fix_npy_file(path, EXPECTED_COCH_SHAPE):
            coch_fixed_count += 1

    print("\n--- 修复完成 ---")
    print(f"总共修复了 {mel_fixed_count} 个梅尔图文件。")
    print(f"总共修复了 {coch_fixed_count} 个耳蜗图文件。")
    print("现在您的数据集长度已经完全一致，可以重新开始训练了！")