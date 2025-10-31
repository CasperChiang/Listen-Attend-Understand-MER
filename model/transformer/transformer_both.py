# -------------------------------------------------------------------
# 文件名: train_with_attention.py (或者你原来的训练文件名)
# 描述: 这是修改后的完整训练脚本，包含了注意力权重的输出。
# -------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
# 假设你的数据加载脚本是正确的
from data_pro.load_dataset_both import train_dataset, val_dataset, test_dataset
import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"找到 {len(gpus)} 个 GPU，将使用: {gpus[0].name}")
    except RuntimeError as e:
        print(e)
else:
    print("未找到 GPU，将使用 CPU 进行训练。")

# --- 检查数据维度 ---
for (mel_batch, coch_batch), labels_batch in train_dataset.take(1):
    print("--- Inspecting a single batch from the dataset ---")
    print("Mel input batch shape:", mel_batch.shape)
    print("Cochleagram input batch shape:", coch_batch.shape)
    # print("Labels batch shape:", labels_batch.shape)

# --- 模型超参数定义 ---
SEQUENCE_LENGTH = 60
MEL_SHAPE = (96, 44, 1)
COCH_SHAPE = (11, 44, 1)
CNN_FILTERS = 32
CNN_OUTPUT_DIM = 32
TRANSFORMER_HEADS = 8
TRANSFORMER_UNITS = 32
TRANSFORMER_LAYERS = 1
MODEL_DIM = 64
DROPOUT_RATE = 0.2


# --- CNN 部分 (与原来相同) ---
def build_mel_cnn(input_shape=(96, 44, 1), output_dim=32):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Mel_CNN')


def build_coch_cnn(input_shape=(11, 44, 1), output_dim=32):
    inp = layers.Input(shape=input_shape)
    x = layers.Conv2D(CNN_FILTERS, (3, 3), activation='relu', padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(CNN_FILTERS * 2, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    x = layers.Flatten()(x)
    out = layers.Dense(output_dim, activation='relu')(x)
    return keras.Model(inputs=inp, outputs=out, name='Coch_CNN')


mel_cnn = build_mel_cnn(MEL_SHAPE, CNN_OUTPUT_DIM)
coch_cnn = build_coch_cnn(COCH_SHAPE, CNN_OUTPUT_DIM)


# --- 位置编码 (与原来相同) ---
class PositionalEncoding(layers.Layer):
    # 添加一个 alpha 参数
    def __init__(self, position, d_model, alpha=0.5, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        # 将 alpha 保存为成员变量
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
        # 核心改动：在相加前乘以 alpha
        scaled_pos_encoding = self.alpha * self.pos_encoding[:, :tf.shape(inputs)[1], :]
        return inputs + scaled_pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model,
            'alpha': self.alpha  # 确保 alpha 可以被保存
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# --- Transformer Encoder (修改) ---
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0, return_attention=False):
    """
    构建一个Transformer Encoder块。
    新增参数 return_attention: 如果为True，则返回注意力权重。
    """
    # Multi-Head Self-Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)

    # 核心改动：设置 return_attention_scores=True 来获取注意力图
    attn_layer = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )
    attention_output, attention_scores = attn_layer(x, x, return_attention_scores=True)

    x = layers.Dropout(dropout)(attention_output)
    res = x + inputs

    # Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    if return_attention:
        return x + res, attention_scores
    return x + res


# --- 完整模型构建 (修改) ---
def build_mer_model_with_attention():
    """
    构建完整的情感识别模型，并额外输出注意力权重。
    """
    # 1. 定义双输入
    mel_input = layers.Input(shape=(SEQUENCE_LENGTH,) + MEL_SHAPE, name="mel_input")
    coch_input = layers.Input(shape=(SEQUENCE_LENGTH,) + COCH_SHAPE, name="coch_input")

    # 2. 并行CNN特征提取
    mel_features = layers.TimeDistributed(mel_cnn, name="time_dist_mel")(mel_input)
    coch_features = layers.TimeDistributed(coch_cnn, name="time_dist_coch")(coch_input)

    # 3. 特征融合
    fused_features = layers.Concatenate(axis=-1, name="fused_features")([mel_features, coch_features])

    # 4. 位置编码
    positional_encoder = PositionalEncoding(SEQUENCE_LENGTH, MODEL_DIM)
    x = positional_encoder(fused_features)

    # 5. Transformer Encoder 栈
    attention_scores = None  # 初始化变量
    for i in range(TRANSFORMER_LAYERS):
        # 只从最后一层获取注意力权重
        if i == TRANSFORMER_LAYERS - 1:
            x, attention_scores = transformer_encoder_block(
                x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                DROPOUT_RATE, return_attention=True)
        else:
            x = transformer_encoder_block(
                x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                DROPOUT_RATE)

    # 6. 输出层
    # 为注意力输出命名
    attention_scores = layers.Lambda(lambda x: x, name='attention_weights')(attention_scores)
    va_output = layers.Dense(2, activation='linear', name="va_output")(x)

    # 7. 构建模型：现在有两个输出
    model = keras.Model(
        inputs=[mel_input, coch_input],
        outputs=[va_output, attention_scores],  # va_output是主输出，attention_scores是辅助输出
        name="Music_Emotion_Transformer_with_Attention"
    )
    return model


# 实例化最终模型
model = build_mer_model_with_attention()
model.summary()


# --- 损失函数和评估指标 (与原来相同) ---
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


# --- 编译模型 (修改) ---
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

# 因为模型有两个输出，我们需要为每个输出指定损失函数。
# 注意力权重输出仅用于评估，不需要损失函数，所以设为 None。
model.compile(
    optimizer=optimizer,
    loss={
        "va_output": ccc_loss,
        "attention_weights": None  # 注意力权重不需要损失
    },
    metrics={
        "va_output": ['mean_absolute_error', ccc_metric]
    }
)

print("\nModel compiled successfully for multi-output!")

# --- 训练模型 (修改) ---
# 回调函数监控的指标名称会因为多输出而改变，需要加上输出层的名字
# 'val_ccc_metric' -> 'val_va_output_ccc_metric'
history = model.fit(
    train_dataset,
    epochs=200,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.ModelCheckpoint("best_model_with_attention.keras", save_best_only=True,
                                        monitor='val_va_output_ccc_metric', mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_va_output_ccc_metric', mode='max',
                                          factor=0.4, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_va_output_ccc_metric', mode='max',
                                      patience=10, restore_best_weights=True)
    ]
)

# --- 训练后保存文件 ---
print("正在保存最终模型到 final_model_with_attention.keras ...")
model.save('../../result/transformer_both.keras')
print("最终模型已保存。")

print("正在保存训练历史到 training_history.npy ...")
np.save('training_history.npy', history.history)
print("训练历史已保存。")