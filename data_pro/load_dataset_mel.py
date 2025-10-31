import tensorflow as tf
import numpy as np
import pandas as pd
import os

# --- 定义模型所需的常量，方便管理 -- -
SEQUENCE_LENGTH = 60
MEL_BINS = 96
TIME_STEPS_PER_SLICE = 44

# 1. 读取元数据
# --- 在这里加入诊断代码 ---
# metadata_path = '../../data/data.csv'
metadata_path = '../data/data.csv'
print(f"--- 开始诊断 ---")
print(f"当前工作目录是: {os.getcwd()}")
print(f"正在尝试从以下路径加载元数据: {os.path.abspath(metadata_path)}")

# 检查文件是否存在
if not os.path.exists(metadata_path):
    print(f"错误：元数据文件未找到！请检查路径是否正确。程序将退出。")
    exit()  # 如果文件不存在，直接退出

metadata = pd.read_csv(metadata_path)

print(f"成功加载元数据，总行数: {len(metadata)}")
if len(metadata) == 0:
    print("警告：元数据文件为空，这将导致数据集为空，程序将不会有任何输出。程序将退出。")
    exit()  # 如果文件为空，直接退出
else:
    print("元数据前5行:")
    print(metadata.head())
print("--------------------")
# --- 诊断代码结束 ---

audio_ids = metadata['audio_id'].values.astype(str)
mel_paths = metadata['mel_spectrogram_path'].values
va_paths = metadata['va_sequence_path'].values

# 创建索引并打乱
indices = np.arange(len(audio_ids))
np.random.seed(42)
np.random.shuffle(indices)

# 应用打乱后的索引
audio_ids = audio_ids[indices]
mel_paths = mel_paths[indices]
va_paths = va_paths[indices]


# 2. 数据加载函数
def load_data(mel_path, va_path):
    # mel = np.load('../' + mel_path.decode('utf-8')).astype(np.float32)
    # va = np.load('../' + va_path.decode('utf-8')).astype(np.float32)
    mel = np.load(mel_path.decode('utf-8')).astype(np.float32)
    va = np.load(va_path.decode('utf-8')).astype(np.float32)
    return mel, va


# 3. 数据预处理与重塑函数
def preprocess_and_reshape(mel, va, audio_id):
    mel_reshaped = tf.reshape(mel, [MEL_BINS, SEQUENCE_LENGTH, TIME_STEPS_PER_SLICE])
    mel_reshaped = tf.transpose(mel_reshaped, perm=[1, 0, 2])
    mel_final = tf.expand_dims(mel_reshaped, axis=-1)

    mel_final = (mel_final - tf.reduce_min(mel_final)) / (tf.reduce_max(mel_final) - tf.reduce_min(mel_final) + 1e-10)

    mel_final = tf.cast(mel_final, tf.float32)
    va = tf.cast(va, tf.float32)

    return mel_final, (va, audio_id)


# 4. 创建 tf.data.Dataset
def create_dataset(mel_paths, va_paths, audio_ids, batch_size=32, shuffle=True):
    # --- 在创建数据集前再次检查输入是否为空 ---
    if len(mel_paths) == 0:
        print(f"警告: 尝试创建一个空的数据集，因为传入的路径列表为空！")
        # 返回一个空的数据集来避免错误，但表明问题所在
        return tf.data.Dataset.from_tensor_slices((
            tf.constant([], dtype=tf.string),
            tf.constant([], dtype=tf.string),
            tf.constant([], dtype=tf.string)
        ))

    dataset = tf.data.Dataset.from_tensor_slices((mel_paths, va_paths, audio_ids))

    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(mel_paths), seed=42)

    dataset = dataset.map(
        lambda mel_path, va_path, audio_id: (
            *tf.numpy_function(load_data, [mel_path, va_path], [tf.float32, tf.float32]),
            audio_id
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(
        lambda mel, va, audio_id: (
            tf.ensure_shape(mel, [MEL_BINS, SEQUENCE_LENGTH * TIME_STEPS_PER_SLICE]),
            tf.ensure_shape(va, [SEQUENCE_LENGTH, 2]),
            tf.ensure_shape(audio_id, [])
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )

    dataset = dataset.map(preprocess_and_reshape, num_parallel_calls=tf.data.AUTOTUNE)

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset


# 5. 创建训练、验证和测试数据集
train_size = int(0.8 * len(audio_ids))
val_size = int(0.1 * len(audio_ids))
test_size = len(audio_ids) - train_size - val_size

train_mel, val_mel, test_mel = np.split(mel_paths, [train_size, train_size + val_size])
train_va, val_va, test_va = np.split(va_paths, [train_size, train_size + val_size])
train_ids, val_ids, test_ids = np.split(audio_ids, [train_size, train_size + val_size])

print(f"\n数据分割情况:")
print(f"训练集大小: {len(train_mel)}")
print(f"验证集大小: {len(val_mel)}")
print(f"测试集大小: {len(test_mel)}")

train_dataset = create_dataset(
    train_mel, train_va, train_ids, batch_size=32
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
val_dataset = create_dataset(
    val_mel, val_va, val_ids, batch_size=32, shuffle=False
).cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = create_dataset(
    test_mel, test_va, test_ids, batch_size=32, shuffle=False
)

# --- 验证最终输出形状 ---
print("\n--- 检查最终数据集输出形状 ---")
# 使用 .cardinality() 检查数据集中的批次数量
train_batches = train_dataset.cardinality()
if train_batches == 0:
    print("最终的 train_dataset 为空，无法迭代。请检查数据源和分割逻辑。")
else:
    print(f"训练集包含 {train_batches.numpy()} 个批次。")
    for mel_batch, (labels_batch, ids_batch) in train_dataset.take(1):
        print("Mel input batch shape:", mel_batch.shape)
        print("Labels batch shape:", labels_batch.shape)
        print("Audio IDs batch shape:", ids_batch.shape)
        print("Example Audio ID:", ids_batch[0].numpy().decode('utf-8'))
