# -------------------------------------------------------------------
# 文件名: evaluate_model.py
# 描述: 用于评估训练好的音乐情感识别模型的脚本。
#       加载模型和测试数据，计算并打印 CCC, R^2, 和 RMSE 指标。
# -------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os

# --- 关键配置 ---
# !!! 请确保这里的路径指向你训练好的模型文件 !!!
MODEL_PATH = '../result/transformer_both.keras'

# 假设你的数据加载脚本是正确的
# 这将从测试集中加载数据
try:
    from data_pro.load_dataset_both import test_dataset
except ImportError:
    print("错误: 无法从 'data_pro.load_dataset_both' 导入 test_dataset。")
    print("请确保 evaluate_model.py 文件与 data_pro 目录处于正确的相对位置。")
    exit()

print("TensorFlow Version:", tf.__version__)


# region Keras 自定义对象定义
# 在加载模型时，Keras 需要知道所有自定义层和函数的定义。
# 因此，我们从训练脚本中复制这些定义。

# --- 位置编码层 ---
class PositionalEncoding(layers.Layer):
    def __init__(self, position, d_model, alpha=0.3, **kwargs):
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
        config.update({
            'position': self.position,
            'd_model': self.d_model,
            'alpha': self.alpha
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# --- CCC 损失/指标函数 ---
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


# endregion

def evaluate():
    """
    主评估函数：加载模型，进行预测，并计算各项指标。
    """
    if not os.path.exists(MODEL_PATH):
        print(f"错误: 模型文件未找到于 '{MODEL_PATH}'")
        return

    # 1. 加载模型
    print(f"正在从 '{MODEL_PATH}' 加载模型...")
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'ccc_loss': ccc_loss,
        'ccc_metric': ccc_metric
    }
    try:
        model = keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
        print("模型加载成功！")
    except Exception as e:
        print(f"加载模型时发生错误: {e}")
        return

    # 2. 逐批次进行预测并收集真实标签和预测结果
    print("\n正在对测试集进行逐批次预测并收集结果...")
    y_true_list = []
    y_pred_list = []

    # 遍历测试数据集的每一个批次
    for (mel_batch, coch_batch), labels_batch in test_dataset:
        # 对当前批次进行预测
        # model.predict_on_batch 比 model.predict 更适合在循环中使用
        predictions = model.predict_on_batch([mel_batch, coch_batch])

        # 模型的输出是一个列表 [va_output, attention_scores]
        # 我们只需要第一个输出
        y_pred_va_batch = predictions[0]

        # 将当前批次的真实标签和预测结果添加到列表中
        y_true_list.append(labels_batch.numpy())
        y_pred_list.append(y_pred_va_batch)

    # 检查是否收集到了数据
    if not y_true_list:
        print("错误：测试数据集为空或无法提取标签。")
        return

    # 3. 将所有批次的结果连接成一个大的Numpy数组
    # 现在 y_true_list 和 y_pred_list 都是 NumPy 数组的列表，可以安全地连接
    y_true = np.concatenate(y_true_list, axis=0)
    y_pred_va = np.concatenate(y_pred_list, axis=0)

    print(f"真实标签数组的总形状: {y_true.shape}")
    print(f"预测值数组的总形状: {y_pred_va.shape}")

    # 4. 预处理标签和预测值以进行评估 (这部分和原来一样)
    num_dims = y_true.shape[-1]
    if num_dims != 2:
        print(f"错误：期望标签的最后一个维度为2（V和A），但得到 {num_dims}")
        return

    y_true_reshaped = y_true.reshape(-1, num_dims)
    y_pred_reshaped = y_pred_va.reshape(-1, num_dims)

    # 分离 Valence 和 Arousal
    y_true_v = y_true_reshaped[:, 0]
    y_true_a = y_true_reshaped[:, 1]
    y_pred_v = y_pred_reshaped[:, 0]
    y_pred_a = y_pred_reshaped[:, 1]

    # 5. 计算各项指标 (这部分和原来一样)
    print("\n--- 模型性能评估结果 ---")

    # --- CCC (Concordance Correlation Coefficient) ---
    ccc_scores = _ccc_per_channel(y_true_reshaped, y_pred_reshaped).numpy()
    ccc_v = ccc_scores[0]
    ccc_a = ccc_scores[1]

    # --- R^2 (R-squared) ---
    r2_v = r2_score(y_true_v, y_pred_v)
    r2_a = r2_score(y_true_a, y_pred_a)

    # --- RMSE (Root Mean Squared Error) ---
    rmse_v = np.sqrt(mean_squared_error(y_true_v, y_pred_v))
    rmse_a = np.sqrt(mean_squared_error(y_true_a, y_pred_a))

    # 6. 打印结果 (这部分和原来一样)
    print("\n[Valence (效价)]")
    print(f"  CCC   : {ccc_v:.4f}")
    print(f"  R^2   : {r2_v:.4f}")
    print(f"  RMSE  : {rmse_v:.4f}")

    print("\n[Arousal (唤醒度)]")
    print(f"  CCC   : {ccc_a:.4f}")
    print(f"  R^2   : {r2_a:.4f}")
    print(f"  RMSE  : {rmse_a:.4f}")
    print("\n--------------------------")


if __name__ == "__main__":
    evaluate()