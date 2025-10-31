# 文件名: evaluation.py (修改后)

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

# 1. 导入你的数据加载器和自定义对象
from data_pro.load_dataset_both import test_dataset

# ==============================================================================
#                      加载模型所需的自定义对象
# ==============================================================================
class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model, alpha=0.1, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.alpha = alpha
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model)
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        scaled_pos_encoding = self.alpha * self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return inputs + scaled_pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'position': self.position, 'd_model': self.d_model, 'alpha': self.alpha})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def _ccc_per_channel(y_true, y_pred, eps=1e-8):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_true = tf.reshape(y_true, [-1, tf.shape(y_true)[-1]])
    y_pred = tf.reshape(y_pred, [-1, tf.shape(y_pred)[-1]])
    true_mean = tf.reduce_mean(y_true, axis=0)
    pred_mean = tf.reduce_mean(y_pred, axis=0)
    true_var = tf.math.reduce_variance(y_true, axis=0)
    pred_var = tf.math.reduce_variance(y_pred, axis=0)
    cov = tf.reduce_mean((y_true - true_mean) * (y_pred - pred_mean), axis=0)
    ccc = (2.0 * cov) / (true_var + pred_var + tf.square(true_mean - pred_mean) + eps)
    return ccc


def ccc_loss(y_true, y_pred):
    ccc = _ccc_per_channel(y_true, y_pred)
    return 1.0 - tf.reduce_mean(ccc)


def ccc_metric(y_true, y_pred):
    ccc = _ccc_per_channel(y_true, y_pred)
    return tf.reduce_mean(ccc)


def plot_attention_heatmap(attention_matrix, title="Attention Heatmap"):
    avg_attention = np.mean(attention_matrix, axis=0)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(avg_attention, cmap='viridis')
    ax.invert_yaxis()
    plt.title(title)
    plt.xlabel('Key (Attended-to Time Step)')
    plt.ylabel('Query (Attending Time Step)')
    plt.show()


# --- 全局参数 ---
SEQUENCE_LENGTH = 60

# 2. 加载模型
custom_objects = {
    "PositionalEncoding": PositionalEncoding,
    "ccc_loss": ccc_loss,
    "ccc_metric": ccc_metric
}
model_path = '../result/transformer_both.keras'
print(f"正在从 {model_path} 加载模型...")
model = keras.models.load_model(model_path, custom_objects=custom_objects)
print("模型加载成功！")

# 3. 加载整个测试集到内存
print("\n正在将整个测试数据集加载到内存中...")
try:
    unbatched_test_data = test_dataset.unbatch()
    all_test_samples = list(unbatched_test_data.as_numpy_iterator())
    print(f"加载完成！测试集中共有 {len(all_test_samples)} 个样本。")
except tf.errors.OutOfRangeError:
    print("数据加载完成。")

# 4. 随机抽样
num_samples_to_show = 10
if len(all_test_samples) < num_samples_to_show:
    num_samples_to_show = len(all_test_samples)

selected_samples = random.sample(all_test_samples, num_samples_to_show)
print(f"\n已随机选取 {num_samples_to_show} 个样本进行可视化。")

# 5. 循环预测和可视化 (核心修改)
# 修改：解包时，同时获取 label 和 audio_id
for i, (features, (label, audio_id)) in enumerate(selected_samples):
    print(f"\n--- 正在处理随机样本 #{i + 1} ---")

    # 修改：将 audio_id 从字节解码为字符串
    # TensorFlow 在 numpy_iterator 中会返回字节字符串 (b'...')
    audio_id_str = audio_id.decode('utf-8')
    print(f"当前样本的 Audio ID: {audio_id_str}")

    # 解包特征
    sample_mel, sample_coch = features

    # 增加批次维度
    sample_mel_batch = np.expand_dims(sample_mel, axis=0)
    sample_coch_batch = np.expand_dims(sample_coch, axis=0)

    # 进行预测
    predictions, attention_weights = model.predict([sample_mel_batch, sample_coch_batch])

    # 提取注意力权重
    attention_to_plot = attention_weights[0]

    # 修改：使用 audio_id_str 作为图表标题
    plot_title = f'Attention Heatmap for Audio ID: {audio_id_str}'
    plot_attention_heatmap(attention_to_plot, title=plot_title)

print("\n--- 所有选定样本的评估已完成 ---")