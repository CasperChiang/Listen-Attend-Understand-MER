import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import os


# ==============================================================================
#                      加载模型所需的自定义对象
# ==============================================================================
# Keras 加载模型时需要知道这些自定义对象的定义。

class PositionalEncoding(keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model

    def build(self, input_shape):
        super(PositionalEncoding, self).build(input_shape)
        self.pos_encoding = self._positional_encoding(self.position, self.d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def _positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, :tf.shape(inputs)[1], :]

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
        })
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


def np_ccc(y_true, y_pred, eps=1e-8):
    """NumPy version of CCC for final evaluation."""
    true_mean = np.mean(y_true)
    pred_mean = np.mean(y_pred)
    true_var = np.var(y_true)
    pred_var = np.var(y_pred)
    cov = np.mean((y_true - true_mean) * (y_pred - pred_mean))
    ccc = (2.0 * cov) / (true_var + pred_var + (true_mean - pred_mean) ** 2 + eps)
    return ccc


# ==============================================================================
#                              主评估函数
# ==============================================================================

def evaluate(model_path):
    """
    加载模型和测试数据，并计算详细的评估指标。
    """
    if not os.path.exists(model_path):
        print(f"错误: 模型文件未找到于 '{model_path}'")
        return

    print("--- 正在加载模型和数据 ---")
    custom_objects = {
        'PositionalEncoding': PositionalEncoding,
        'ccc_loss': ccc_loss,
        'ccc_metric': ccc_metric
    }
    model = keras.models.load_model(model_path, custom_objects=custom_objects)
    print(f"模型 '{model_path}' 加载成功。")

    try:
        from load_dataset import test_dataset
        print("测试数据集加载成功。")
    except ImportError:
        print("错误: 无法导入 'test_dataset'。请确保 'load_dataset.py' 文件在当前目录中。")
        return

    print("\n--- 正在生成预测 ---")
    all_predictions, all_labels = [], []
    for (mel_batch, coch_batch), labels_batch in test_dataset:
        predictions_batch = model.predict_on_batch((mel_batch, coch_batch))
        all_predictions.append(predictions_batch)
        all_labels.append(labels_batch.numpy())

    y_pred = np.concatenate(all_predictions, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
    y_true_flat = y_true.reshape(-1, y_true.shape[-1])
    print(f"已处理完所有测试样本，总时间步数: {y_pred_flat.shape[0]}")

    print("\n--- 正在计算评估指标 ---")
    y_true_v, y_pred_v = y_true_flat[:, 0], y_pred_flat[:, 0]
    y_true_a, y_pred_a = y_true_flat[:, 1], y_pred_flat[:, 1]

    # 计算 Valence 指标
    ccc_v, mae_v = np_ccc(y_true_v, y_pred_v), np.mean(np.abs(y_true_v - y_pred_v))
    rmse_v, r2_v = np.sqrt(mean_squared_error(y_true_v, y_pred_v)), r2_score(y_true_v, y_pred_v)

    # 计算 Arousal 指标
    ccc_a, mae_a = np_ccc(y_true_a, y_pred_a), np.mean(np.abs(y_true_a - y_pred_a))
    rmse_a, r2_a = np.sqrt(mean_squared_error(y_true_a, y_pred_a)), r2_score(y_true_a, y_pred_a)

    # 计算 VA 整体指标
    ccc_overall = (ccc_v + ccc_a) / 2.0  # 新增
    mae_overall = np.mean(np.abs(y_true_flat - y_pred_flat))  # 新增
    rmse_overall = np.sqrt(mean_squared_error(y_true_flat, y_pred_flat))
    r2_overall = r2_score(y_true_flat, y_pred_flat, multioutput='uniform_average')

    # --- 打印报告 ---
    print("\n" + "=" * 50)
    print(" " * 15 + "模型评估报告")
    print("=" * 50)
    print(f"\n模型文件: {model_path}\n")
    print("--- 单独评估结果 (Individual Metrics) ---")
    print(f"Valence - CCC:   {ccc_v:.4f}")
    print(f"Valence - MAE:   {mae_v:.4f}")
    print(f"Valence - RMSE:  {rmse_v:.4f}")
    print(f"Valence - R²:    {r2_v:.4f}")
    print("-" * 50)
    print(f"Arousal - CCC:   {ccc_a:.4f}")
    print(f"Arousal - MAE:   {mae_a:.4f}")
    print(f"Arousal - RMSE:  {rmse_a:.4f}")
    print(f"Arousal - R²:    {r2_a:.4f}")
    print("\n--- 整体评估结果 (Overall Metrics) ---")
    print(f"Overall - CCC:   {ccc_overall:.4f} (average of V and A)")  # 新增
    print(f"Overall - MAE:   {mae_overall:.4f}")  # 新增
    print(f"Overall - RMSE:  {rmse_overall:.4f}")
    print(f"Overall - R²:    {r2_overall:.4f} (uniform average)")
    print("=" * 50)


if __name__ == '__main__':
    # --- 在这里直接指定要评估的模型文件 ---
    MODEL_TO_EVALUATE = 'final_model_R2.keras'

    print(f"准备评估模型: {MODEL_TO_EVALUATE}")
    evaluate(MODEL_TO_EVALUATE)