# -------------------------------------------------------------------
# 文件名: transformer_mel.py
# 描述: 这是一个仅使用Mel谱图进行训练的Transformer模型脚本。
# -------------------------------------------------------------------

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras.models import load_model
# 从新的数据加载脚本导入
from data_pro.load_dataset_mel import train_dataset, val_dataset, test_dataset

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
for mel_batch, (labels_batch, _) in train_dataset.take(1):
    print("--- Inspecting a single batch from the dataset ---")
    print("Mel input batch shape:", mel_batch.shape)
    print("Labels batch shape:", labels_batch.shape)

# --- 模型超参数定义 ---
SEQUENCE_LENGTH = 60
MEL_SHAPE = (96, 44, 1)
# COCH_SHAPE is removed
CNN_FILTERS = 32
CNN_OUTPUT_DIM = 32
TRANSFORMER_HEADS = 8
TRANSFORMER_UNITS = 32
TRANSFORMER_LAYERS = 1
# MODEL_DIM is now CNN_OUTPUT_DIM since we only have one input branch
MODEL_DIM = 32
DROPOUT_RATE = 0.2


# --- CNN 部分 (只保留Mel CNN) ---
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

mel_cnn = build_mel_cnn(MEL_SHAPE, CNN_OUTPUT_DIM)

# --- 位置编码 (与原来相同) ---
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

# --- Transformer Encoder (与原来相同) ---
def transformer_encoder_block(inputs, head_size, num_heads, ff_dim, dropout=0, return_attention=False):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    attn_layer = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )
    attention_output, attention_scores = attn_layer(x, x, return_attention_scores=True)
    x = layers.Dropout(dropout)(attention_output)
    res = x + inputs

    x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)

    if return_attention:
        return x + res, attention_scores
    return x + res


# --- 完整模型构建 (修改为仅Mel输入) ---
def build_mer_model_mel_with_attention():
    '''
    构建仅使用Mel谱图的情感识别模型，并额外输出注意力权重。
    '''
    # 1. 定义单输入
    mel_input = layers.Input(shape=(SEQUENCE_LENGTH,) + MEL_SHAPE, name="mel_input")

    # 2. CNN特征提取
    mel_features = layers.TimeDistributed(mel_cnn, name="time_dist_mel")(mel_input)

    # 3. 位置编码
    positional_encoder = PositionalEncoding(SEQUENCE_LENGTH, MODEL_DIM)
    x = positional_encoder(mel_features) # Directly use mel_features

    # 4. Transformer Encoder 栈
    attention_scores = None
    for i in range(TRANSFORMER_LAYERS):
        if i == TRANSFORMER_LAYERS - 1:
            x, attention_scores = transformer_encoder_block(
                x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                DROPOUT_RATE, return_attention=True)
        else:
            x = transformer_encoder_block(
                x, MODEL_DIM // TRANSFORMER_HEADS, TRANSFORMER_HEADS, TRANSFORMER_UNITS,
                DROPOUT_RATE)

    # 5. 输出层
    attention_scores = layers.Lambda(lambda x: x, name='attention_weights')(attention_scores)
    va_output = layers.Dense(2, activation='linear', name="va_output")(x)

    # 6. 构建模型
    model = keras.Model(
        inputs=mel_input, # Single input
        outputs=[va_output, attention_scores],
        name="Music_Emotion_Transformer_Mel_with_Attention"
    )
    return model


# 实例化最终模型
model = build_mer_model_mel_with_attention()
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


# --- 编译模型 (与原来相同) ---
optimizer = keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

model.compile(
    optimizer=optimizer,
    loss={
        "va_output": ccc_loss,
        "attention_weights": None
    },
    metrics={
        "va_output": ['mean_absolute_error', ccc_metric]
    }
)

print("\nModel compiled successfully for multi-output!")

# --- 训练模型 (回调函数监控指标名称已更新) ---
history = model.fit(
    train_dataset, # The dataset now yields (mel_batch, (labels, ids))
    epochs=200,
    validation_data=val_dataset,
    callbacks=[
        keras.callbacks.ModelCheckpoint("best_model_mel_with_attention.keras", save_best_only=True,
                                        monitor='val_va_output_ccc_metric', mode='max'),
        keras.callbacks.ReduceLROnPlateau(monitor='val_va_output_ccc_metric', mode='max',
                                          factor=0.4, patience=3, min_lr=1e-7, verbose=1),
        keras.callbacks.EarlyStopping(monitor='val_va_output_ccc_metric', mode='max',
                                      patience=6, restore_best_weights=True)
    ]
)

# --- 训练后保存文件 ---
print("正在保存最终模型到 final_model_mel_with_attention.keras ...")
model.save('../../result/transformer_mel.keras')
print("最终模型已保存。")

print("正在保存训练历史到 training_history_mel.npy ...")
model.save('training_history_mel.npy', history.history)
print("训练历史已保存。")
