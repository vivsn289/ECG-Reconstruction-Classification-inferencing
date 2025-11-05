"""
MobileNetV2-inspired CNN architecture with ECA attention
Optimized for ECG image classification (~500K parameters)
"""

import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from src import config


class ECA(layers.Layer):
    """Efficient Channel Attention mechanism"""
    
    def __init__(self, gamma=2, b=1, **kwargs):
        super(ECA, self).__init__(**kwargs)
        self.gamma = gamma
        self.b = b
    
    def build(self, input_shape):
        channels = input_shape[-1]
        t = int(abs(math.log2(channels) / self.b + self.gamma))
        if t % 2 == 0:
            t += 1
        self.k_size = t
        
        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.conv = layers.Conv1D(1, kernel_size=self.k_size, 
                                  padding='same', use_bias=False)
        self.sigmoid = layers.Activation('sigmoid')
        super().build(input_shape)
    
    def call(self, x):
        y = self.avg_pool(x)
        y = tf.transpose(y, [0, 3, 1, 2])
        y = tf.squeeze(y, [2, 3])
        y = tf.expand_dims(y, -1)
        y = self.conv(y)
        y = self.sigmoid(y)
        y = tf.expand_dims(tf.transpose(y, [0, 2, 1]), 1)
        return x * y
    
    def get_config(self):
        config_dict = super().get_config()
        config_dict.update({'gamma': self.gamma, 'b': self.b})
        return config_dict


def inverted_residual_block(x, out_channels, stride, expansion, block_id):
    """
    MobileNetV2 inverted residual block with ECA attention
    
    Args:
        x: Input tensor
        out_channels: Number of output channels
        stride: Stride for depthwise convolution
        expansion: Expansion factor for intermediate channels
        block_id: Block identifier for naming
        
    Returns:
        Output tensor
    """
    in_channels = x.shape[-1]
    prefix = f'block_{block_id}_'
    
    # Expansion phase
    if expansion != 1:
        x_exp = layers.Conv2D(in_channels * expansion, (1, 1), 
                             padding='same', use_bias=False,
                             name=prefix + 'expand')(x)
        x_exp = layers.BatchNormalization(name=prefix + 'expand_bn')(x_exp)
        x_exp = layers.ReLU(6., name=prefix + 'expand_relu')(x_exp)
    else:
        x_exp = x
    
    # Depthwise convolution
    x_dw = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same',
                                 use_bias=False, name=prefix + 'depthwise')(x_exp)
    x_dw = layers.BatchNormalization(name=prefix + 'depthwise_bn')(x_dw)
    x_dw = layers.ReLU(6., name=prefix + 'depthwise_relu')(x_dw)
    
    # ECA attention
    x_att = ECA(name=prefix + 'eca')(x_dw)
    
    # Projection phase
    x_proj = layers.Conv2D(out_channels, (1, 1), padding='same',
                          use_bias=False, name=prefix + 'project')(x_att)
    x_proj = layers.BatchNormalization(name=prefix + 'project_bn')(x_proj)
    
    # Residual connection
    if stride == 1 and in_channels == out_channels:
        x_out = layers.Add(name=prefix + 'add')([x, x_proj])
    else:
        x_out = x_proj
    
    return x_out


def create_model(input_shape=(512, 512, 3), num_classes=5):
    """
    Create MobileNet student model with ECA attention
    
    Args:
        input_shape: Input image shape
        num_classes: Number of output classes
        
    Returns:
        Compiled Keras model
    """
    inputs = layers.Input(shape=input_shape, name='input')
    
    # Stem convolution
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same',
                     use_bias=False, name='stem_conv')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(6., name='stem_relu')(x)
    
    # Bottleneck configuration
    bottleneck_configs = [
        (16,  1, 1, 1),   # (filters, stride, expansion, repeats)
        (24,  2, 6, 1),
        (32,  2, 6, 2),
        (64,  2, 6, 2),
        (96,  1, 6, 2),
        (160, 2, 6, 1),
    ]
    
    block_id = 0
    for filters, stride, expansion, num_repeats in bottleneck_configs:
        for i in range(num_repeats):
            s = stride if i == 0 else 1
            x = inverted_residual_block(x, filters, s, expansion, block_id)
            block_id += 1
    
    # Final convolution
    x = layers.Conv2D(640, (1, 1), padding='same', 
                     use_bias=False, name='final_conv')(x)
    x = layers.BatchNormalization(name='final_bn')(x)
    x = layers.ReLU(6., name='final_relu')(x)
    
    # Classification head
    x = layers.GlobalAveragePooling2D(name='global_pool')(x)
    x = layers.Dropout(config.DROPOUT_RATE, name='dropout')(x)
    x = layers.Dense(num_classes, dtype='float32', 
                    name='predictions_dense')(x)
    outputs = layers.Activation('softmax', dtype='float32', 
                                name='predictions')(x)
    
    model = models.Model(inputs, outputs, name='MobileLiteNet_ECG')
    
    return model


if __name__ == "__main__":
    # Test model creation
    model = create_model(
        input_shape=(config.IMG_SIZE, config.IMG_SIZE, 3),
        num_classes=config.NUM_CLASSES
    )
    
    print(f"✓ Model created")
    print(f"  Total parameters: {model.count_params():,}")
    print(f"  Input shape: {model.input_shape}")
    print(f"  Output shape: {model.output_shape}")
    
    model.summary()
