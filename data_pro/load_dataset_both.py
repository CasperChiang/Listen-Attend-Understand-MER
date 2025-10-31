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
# metadata = pd.read_csv('../../data/data.csv')
metadata = pd.read_csv('../data/data.csv')
audio_ids = metadata['audio_id'].values.astype(str)
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
    # coch = np.load('../' + coch_path.decode('utf-8')).astype(np.float32)
    # mel = np.load('../' + mel_path.decode('utf-8')).astype(np.float32)
    # va = np.load('../' + va_path.decode('utf-8')).astype(np.float32)

    coch = np.load(coch_path.decode('utf-8')).astype(np.float32)
    mel = np.load(mel_path.decode('utf-8')).astype(np.float32)
    va = np.load(va_path.decode('utf-8')).astype(np.float32)
    return coch, mel, va

# 3. 数据预处理与重塑函数 (***核心修改***)
def preprocess_and_reshape(coch, mel, va, audio_id):
    # --- 特征重塑部分保持不变 ---
    mel_reshaped = tf.reshape(mel, [MEL_BINS, SEQUENCE_LENGTH, TIME_STEPS_PER_SLICE])
    mel_reshaped = tf.transpose(mel_reshaped, perm=[1, 0, 2])
    coch_reshaped = tf.reshape(coch, [COCH_CHANNELS, SEQUENCE_LENGTH, TIME_STEPS_PER_SLICE])
    coch_reshaped = tf.transpose(coch_reshaped, perm=[1, 0, 2])
    mel_final = tf.expand_dims(mel_reshaped, axis=-1)
    coch_final = tf.expand_dims(coch_reshaped, axis=-1)

    # --- 归一化部分保持不变 ---
    coch_final = (coch_final - tf.reduce_min(coch_final)) / (
        tf.reduce_max(coch_final) - tf.reduce_min(coch_final) + 1e-10)
    mel_final = (mel_final - tf.reduce_min(mel_final)) / (tf.reduce_max(mel_final) - tf.reduce_min(mel_final) + 1e-10)

    # --- 类型转换部分保持不变 ---
    coch_final = tf.cast(coch_final, tf.float32)
    mel_final = tf.cast(mel_final, tf.float32)
    va = tf.cast(va, tf.float32)

    # 修改：返回的标签现在是一个元组 (va, audio_id)
    return (mel_final, coch_final), (va, audio_id)

# 4. 创建 tf.data.Dataset (修改：处理 audio_id)
def create_dataset(coch_paths, mel_paths, va_paths, audio_ids, batch_size=32, shuffle=True):
    # 修改：在 from_tensor_slices 中加入 audio_ids
    dataset = tf.data.Dataset.from_tensor_slices((coch_paths, mel_paths, va_paths, audio_ids))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(coch_paths), seed=42)

    # 修改：让 map 函数能正确传递 audio_id
    dataset = dataset.map(
        lambda coch_path, mel_path, va_path, audio_id: (
            *tf.numpy_function(load_data, [coch_path, mel_path, va_path], [tf.float32, tf.float32, tf.float32]),
            audio_id
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 修改：为 audio_id 也设置形状（标量字符串，形状为()）
    dataset = dataset.map(
        lambda coch, mel, va, audio_id: (
            tf.ensure_shape(coch, [COCH_CHANNELS, SEQUENCE_LENGTH * TIME_STEPS_PER_SLICE]),
            tf.ensure_shape(mel, [MEL_BINS, SEQUENCE_LENGTH * TIME_STEPS_PER_SLICE]),
            tf.ensure_shape(va, [SEQUENCE_LENGTH, 2]),
            tf.ensure_shape(audio_id, [])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # 修改：将 audio_id 传递给预处理函数
    dataset = dataset.map(preprocess_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)

    # 批处理
    dataset = dataset.batch(batch_size)
    # 预取
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset

# 5. 创建训练、验证和测试数据集 (修改：切分并传递 audio_ids)
train_size = int(0.8 * len(audio_ids))
val_size = int(0.1 * len(audio_ids))
test_size = len(audio_ids) - train_size - val_size

# 修改：同时切分 audio_ids
train_coch, val_coch, test_coch = np.split(coch_paths, [train_size, train_size + val_size])
train_mel, val_mel, test_mel = np.split(mel_paths, [train_size, train_size + val_size])
train_va, val_va, test_va = np.split(va_paths, [train_size, train_size + val_size])
train_ids, val_ids, test_ids = np.split(audio_ids, [train_size, train_size + val_size]) # 新增

# 修改：将 ids 传递给 create_dataset 函数
train_dataset = create_dataset(
    train_coch, train_mel, train_va, train_ids, batch_size=32
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = create_dataset(
    val_coch, val_mel, val_va, val_ids, batch_size=32, shuffle=False
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = create_dataset(
    test_coch, test_mel, test_va, test_ids, batch_size=32, shuffle=False
)

# --- 验证最终输出形状 ---
print("--- 检查最终数据集输出形状 ---")
for (mel_batch, coch_batch), (labels_batch, ids_batch) in train_dataset.take(1):
    print("Mel input batch shape:", mel_batch.shape)
    print("Cochleagram input batch shape:", coch_batch.shape)
    print("Labels batch shape:", labels_batch.shape)
    print("Audio IDs batch shape:", ids_batch.shape) # 新增
    print("Example Audio ID:", ids_batch[0].numpy().decode('utf-8')) # 新增