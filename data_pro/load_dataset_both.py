import tensorflow as tf
import numpy as np
import pandas as pd
import os

# --- 定义模型所需的常量，方便管理 ---
SEQUENCE_LENGTH = 60
MEL_BINS = 96
COCH_CHANNELS = 11
TIME_STEPS_PER_SLICE = 44

# 1. 读取元数据 (这部分保持不变)
metadata = pd.read_csv('../data/data.csv')
audio_ids = metadata['audio_id'].values
coch_paths = metadata['cochleagram_path'].values
mel_paths = metadata['mel_spectrogram_path'].values
va_paths = metadata['va_sequence_path'].values

# 创建索引并打乱
indices = np.arange(len(audio_ids))
np.random.seed(42)
np.random.shuffle(indices)

# 应用打乱后的索引
audio_ids = audio_ids[indices]
coch_paths = coch_paths[indices]
mel_paths = mel_paths[indices]
va_paths = va_paths[indices]

# 2. 数据加载函数 (这部分保持不变)
def load_data(coch_path, mel_path, va_path):
    coch = np.load(coch_path.decode('utf-8')).astype(np.float32)
    mel = np.load(mel_path.decode('utf-8')).astype(np.float32)
    va = np.load(va_path.decode('utf-8')).astype(np.float32)
    return coch, mel, va

# 3. 数据预处理与重塑函数 (***核心修改***)
def preprocess_and_reshape(coch, mel, va):
    # --- 关键步骤：将扁平化的数据重塑为模型需要的形状 ---
    # 根据之前的调试信息，coch形状是(11, 2640), mel是(96, 2640)

    # a. 重塑梅尔图 (Mel)
    # (96, 2640) -> (96, 60, 44)
    mel_reshaped = tf.reshape(mel, [MEL_BINS, SEQUENCE_LENGTH, TIME_STEPS_PER_SLICE])
    # 交换维度以匹配模型输入 (序列, 高, 宽): (96, 60, 44) -> (60, 96, 44)
    mel_reshaped = tf.transpose(mel_reshaped, perm=[1, 0, 2])

    # b. 重塑耳蜗图 (Cochleagram)
    # (11, 2640) -> (11, 60, 44)
    coch_reshaped = tf.reshape(coch, [COCH_CHANNELS, SEQUENCE_LENGTH, TIME_STEPS_PER_SLICE])
    # 交换维度: (11, 60, 44) -> (60, 11, 44)
    coch_reshaped = tf.transpose(coch_reshaped, perm=[1, 0, 2])

    # c. 为CNN增加通道维度
    mel_final = tf.expand_dims(mel_reshaped, axis=-1) # -> (60, 96, 44, 1)
    coch_final = tf.expand_dims(coch_reshaped, axis=-1) # -> (60, 11, 44, 1)

    # d. 归一化 (示例：全局 min-max 归一化)
    # 注意：更好的做法是计算训练集的全局均值和标准差进行标准化
    coch_final = (coch_final - tf.reduce_min(coch_final)) / (
        tf.reduce_max(coch_final) - tf.reduce_min(coch_final) + 1e-10)
    mel_final = (mel_final - tf.reduce_min(mel_final)) / (tf.reduce_max(mel_final) - tf.reduce_min(mel_final) + 1e-10)

    # e. 确保数据类型
    coch_final = tf.cast(coch_final, tf.float32)
    mel_final = tf.cast(mel_final, tf.float32)
    va = tf.cast(va, tf.float32)

    return (mel_final, coch_final), va

# 4. 创建 tf.data.Dataset (***优化修改***)
def create_dataset(coch_paths, mel_paths, va_paths, batch_size=32, shuffle=True):
    dataset = tf.data.Dataset.from_tensor_slices((coch_paths, mel_paths, va_paths))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(coch_paths), seed=42)

    # 加载数据
    dataset = dataset.map(
        lambda coch_path, mel_path, va_path: tf.numpy_function(
            load_data, [coch_path, mel_path, va_path], [tf.float32, tf.float32, tf.float32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # ***新增：为 tf.numpy_function 的输出设置形状***
    # 这一步至关重要，它告诉 TensorFlow 图加载后的数据形状，避免"未知形状"错误
    dataset = dataset.map(
        lambda coch, mel, va: (
            tf.ensure_shape(coch, [COCH_CHANNELS, SEQUENCE_LENGTH * TIME_STEPS_PER_SLICE]),
            tf.ensure_shape(mel, [MEL_BINS, SEQUENCE_LENGTH * TIME_STEPS_PER_SLICE]),
            tf.ensure_shape(va, [SEQUENCE_LENGTH, 2])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 应用预处理和重塑
    dataset = dataset.map(preprocess_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)

    # ***新增：数据批处理***
    # model.fit 需要的是批处理数据
    dataset = dataset.batch(batch_size)

    # 预取数据以提高性能
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# 5. 创建训练和验证数据集 (***修正切分逻辑***)
train_size = int(0.8 * len(audio_ids))
val_size = int(0.1 * len(audio_ids)) # 例如，10% 用于验证
test_size = len(audio_ids) - train_size - val_size # 剩余用于测试

# 修正切片索引
train_coch, val_coch, test_coch = np.split(coch_paths, [train_size, train_size + val_size])
train_mel, val_mel, test_mel = np.split(mel_paths, [train_size, train_size + val_size])
train_va, val_va, test_va = np.split(va_paths, [train_size, train_size + val_size])

train_dataset = create_dataset(
    train_coch, train_mel, train_va, batch_size=32
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = create_dataset(
    val_coch, val_mel, val_va, batch_size=32, shuffle=False
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = create_dataset(
    test_coch, test_mel, test_va, batch_size=32, shuffle=False
)

# --- 验证一下最终输出的形状 ---
print("--- 检查最终数据集输出形状 ---")
for (mel_batch, coch_batch), labels_batch in train_dataset.take(1):
    print("Mel input batch shape:", mel_batch.shape)
    print("Cochleagram input batch shape:", coch_batch.shape)
    print("Labels batch shape:", labels_batch.shape)