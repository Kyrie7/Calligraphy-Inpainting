import tensorflow as tf
import os
import time

# Instance Normalization
class InstanceNormalization(tf.keras.layers.Layer):
	def __init__(self, epsilon=1e-5):
		super(InstanceNormalization, self).__init__()
		self.epsilon = epsilon
	
	def build(self, input_shape):
		self.scale = self.add_weight(
			name='scale',
			shape=input_shape[-1:],
			initializer=tf.random_normal_initializer(1., 0.02),
			trainable=True)
		self.offset = self.add_weight(
			name='offset',
			shape=input_shape[-1:],
			initializer='zeros',
			trainable=True)
	
	def call(self, x):
		mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
		inv = tf.math.rsqrt(variance + self.epsilon)
		normalized = (x - mean) * inv
		return normalized * self.scale + self.offset

# downsample block
def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
	"""Downsample an input.

	Conv2D => Instancenorm => LeakyReLU

	Args:
		filters: number of filters
		size: filter size
		norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
		apply_norm: If True, adds the norm layer

	Returns:
		Downsample Sequential Model
	"""

	initializer = tf.random_normal_initializer(0., 0.02)

	block = tf.keras.Sequential()

	block.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
					kernel_initializer=initializer,
					use_bias=False))
	if apply_norm:
		if norm_type.lower() == 'batchnorm':
			block.add(tf.keras.layers.BatchNormalization())
		elif norm_type.lower() == 'instancenorm':
			block.add(InstanceNormalization())
	block.add(tf.keras.layers.LeakyReLU(0.2))

	return block

def residual_block_down(inputs, filters, strides=(1, 1), norm_type='batchnorm'):
	if strides == (1, 1):
		shortcut = inputs
	else:
		# decrease length and width of inputs
		shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides)(inputs)

	# first conv2d, may decrease the size
	net = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
	if norm_type.lower() == 'batchnorm':
		net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	elif norm_type.lower() == 'instancenorm':
		net = InstanceNormalization()(net)
	net = tf.keras.layers.ReLU()(net)

	# second conv2d, do not decrease the size
	net = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
	if norm_type.lower() == 'batchnorm':
		net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	elif norm_type.lower() == 'instancenorm':
		net = InstanceNormalization()(net)
	net = tf.keras.layers.Add()([net, shortcut])
	net = tf.keras.layers.ReLU()(net)

	return net

# upsample block
def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
	"""Upsample an input.

	Conv2DTranspose => Norm => Dropout => ReLU

	Args:
		filters: number of filters
		size: filter size
		norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
		apply_dropout: If True, adds the dropout layer

	Returns:
		Upsample Sequential Model
	"""
	initializer = tf.random_normal_initializer(0., 0.02)

	block = tf.keras.Sequential()

	block.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same',
							kernel_initializer=initializer,
							use_bias=False))
	if norm_type.lower() == 'batchnorm':
		block.add(tf.keras.layers.BatchNormalization())
	elif norm_type.lower() == 'instancenorm':
		block.add(InstanceNormalization())

	if apply_dropout:
		block.add(tf.keras.layers.Dropout(0.5))

	block.add(tf.keras.layers.ReLU())

	return block

def residual_block_up(inputs, filters, strides=(1, 1), norm_type='instancenorm'):
	if strides == (1, 1):
		shortcut = inputs
	else:
		# decrease length and width of inputs
		shortcut = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(1, 1), strides=strides)(inputs)

	# first conv2dTranspose, may increase the size
	net = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
	if norm_type.lower() == 'batchnorm':
		net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	elif norm_type.lower() == 'instancenorm':
		net = InstanceNormalization()(net)
	net = tf.keras.layers.ReLU()(net)

	# second conv2d, do not decrease the size
	net = tf.keras.layers.Conv2DTranspose(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
	if norm_type.lower() == 'batchnorm':
		net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	elif norm_type.lower() == 'instancenorm':
		net = InstanceNormalization()(net)
	net = tf.keras.layers.Add()([net, shortcut])
	net = tf.keras.layers.ReLU()(net)

	return net

"""
Generator

The architecture of Generator is a Encoder-Decoder.

Encoder encoder: C64-C128-C256-C512-C512

All ReLUs in the encoder are leaky, with slope 0.2

So Each block in the encoder is (Conv -> instancenorm -> LeakyReLU)

Encoder first block C64 do not use Norm

Decoder decoder: -C512-C256-C128-C64

ReLUs in the decoder are not leaky

So Each block in the Decoder is (Transposed Conv -> instancenorm -> Dropout(applied to the first 3 blocks) -> ReLU)

After the last layer in the decoder, a convolution is applied to map to the number of output channels.

"""
def Encoder_Decoder(norm_type='instancenorm'):
	""" Encoder-Decoder Generator model

	Args:
		norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.

	Return:
		Generator model
	"""
	inputs = tf.keras.layers.Input(shape=[224, 224, 3])

	down_stack = [
		downsample(64, 4, norm_type, apply_norm=False), # (bs, 112, 112, 64)
		downsample(128, 4, norm_type), # (bs, 56, 56, 128)
		downsample(256, 4, norm_type), # (bs, 28, 28, 256)
		downsample(512, 4, norm_type), # (bs, 14, 14, 512)
		downsample(512, 4, norm_type), # (bs, 7, 7, 512)
	]

	up_stack = [
		upsample(512, 4, norm_type, apply_dropout=True), # (bs, 14, 14, 512)
		upsample(256, 4, norm_type, apply_dropout=True), # (bs, 28, 28, 256)
		upsample(128, 4, norm_type, apply_dropout=True), # (bs, 56, 56, 128)
		upsample(64, 4, norm_type), # (bs, 112, 112, 64)
	]

	# last layer
	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
										kernel_initializer=initializer,
										activation='sigmoid') # (bs, 224, 224, 3)
	x = inputs
	
	for down in down_stack:
		x = down(x)
	
	for up in up_stack:
		x = up(x)

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)


"""
U-Net Generator Model

The architecture of Generator is a modified U-Net.

Encoder encoder: C64-C128-C256-C512-C512

All ReLUs in the encoder are leaky, with slope 0.2

So Each block in the encoder is (Conv -> instancenorm -> Leaky ReLU)

encoder first block C64 do not use Norm

Decoder decoder: -C512-C256-C128-C64

ReLUs in the decoder are not leaky

So Each block in the decoder is (Transposed Conv -> instancenorm -> Dropout(applied to the first 3 blocks) -> ReLU)

After the last layer in the decoder, a convolution is applied to map to the number of output channels (3 in general, except in colorization, where it is 2),

The skip connections concatenate activations from layer i to layer n − i.
"""

def Unet_Generator(norm_type='instancenorm'):
	"""Modified u-net generator model.
    
    Args:
        output_channels: Output channels
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    
    Returns:
        Generator model
	"""

	inputs = tf.keras.layers.Input(shape=[224, 224, 3])

	down_stack = [
        downsample(64, 4, norm_type, apply_norm=False), # (bs, 112, 112, 64)
        downsample(128, 4, norm_type), # (bs, 56, 56, 128)
        downsample(256, 4, norm_type), # (bs, 28, 28, 256)
        downsample(512, 4, norm_type), # (bs, 14, 14, 512)
        downsample(512, 4, norm_type), # (bs, 7, 7, 512)
    ]
	up_stack = [
        upsample(512, 4, norm_type, apply_dropout=True), # (bs, 14, 14, 512)
        upsample(256, 4, norm_type, apply_dropout=True), # (bs, 28, 28, 256)
        upsample(128, 4, norm_type, apply_dropout=True), # (bs, 56, 56, 128)
        upsample(64, 4, norm_type), # (bs, 112, 112, 64)
    ]
    
    # 最后一层
	initializer = tf.random_normal_initializer(0., 0.02)
	last = tf.keras.layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                          kernel_initializer=initializer,
                                          activation='sigmoid') # (bs, 224, 224, 3)
    
	x = inputs
	skips = []
	for down in down_stack:
		x = down(x)
		skips.append(x)
	skips = reversed(skips[:-1])

	for up, skip in zip(up_stack, skips):
		x = up(x)
		x = tf.keras.layers.Concatenate()([x, skip])

	x = last(x)

	return tf.keras.Model(inputs=inputs, outputs=x)


"""
ResNet Generator Model Without Skip Connection

The architecture of Generator is a modified ResNet.

"""

def ResNet_Generator(norm_type='instancenorm'):
	"""Modified resnet generator model without skip-connection
    
    Args:
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    
    Returns:
        Generator model
	"""

	inputs = tf.keras.layers.Input(shape=[224, 224, 3])
	# skips = []
	
	net = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs) # (bs, 112, 112, 64)
	net = InstanceNormalization()(net)
	net = tf.keras.layers.ReLU()(net)
	# skips.append(net)

	net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net) # (bs, 56, 56, 64)
	# skips.append(net)
	
	for i in range(3):
		net = residual_block_down(net, 64, norm_type=norm_type) # (bs, 56, 56, 64)

	# change dimentions
	net = residual_block_down(net, 128, strides=(2, 2), norm_type=norm_type) # (bs, 28, 28, 128)
	# skips.append(net)

	for i in range(3):
		net = residual_block_down(net, 128, norm_type=norm_type) # (bs, 28, 28, 128)

	# change dimentions
	net = residual_block_down(net, 256, strides=(2, 2), norm_type=norm_type) # (bs, 14, 14, 256)
	# skips.append(net)

	for i in range(5):
		net = residual_block_down(net, 256, norm_type=norm_type) # (bs, 14, 14, 256)

	# change dimentions
	net = residual_block_down(net, 512, strides=(2, 2), norm_type=norm_type) # (bs, 7, 7, 512)
	# skips.append(net)

	for i in range(2):
		net = residual_block_down(net, 512, norm_type=norm_type) # (bs, 7, 7, 512)

	# skips = list(reversed(skips[:]))

	for i in range(2):
		net = residual_block_up(net, 512) # (bs, 7, 7, 512)

	# net = tf.keras.layers.Concatenate()([net, skips[0]])

	net = residual_block_up(net, 256, strides=(2, 2), norm_type=norm_type) # (bs, 14, 14, 256)

	for i in range(5):
		net = residual_block_up(net, 256, norm_type=norm_type) # (bs, 14, 14, 256)

	# net = tf.keras.layers.Concatenate()([net, skips[1]])

	net = residual_block_up(net, 128, strides=(2, 2), norm_type=norm_type) # (bs, 28, 28, 128)

	for i in range(3):
		net = residual_block_up(net, 128, norm_type=norm_type) # (bs, 28, 28, 128)

	# net = tf.keras.layers.Concatenate()([net, skips[2]])

	net = residual_block_up(net, 64, strides=(2, 2), norm_type=norm_type) # (bs, 56, 56, 64)

	for i in range(3):
		net = residual_block_up(net, 64, norm_type=norm_type) # (bs, 56, 56, 64)

	# net = tf.keras.layers.Concatenate()([net, skips[3]])

	net = tf.keras.layers.UpSampling2D()(net) # (112, 112, 64)

	# net = tf.keras.layers.Concatenate()([net, skips[4]])
	
	net = tf.keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same',
	activation='sigmoid')(net) # (bs, 224, 224, 3)
	
	return tf.keras.Model(inputs=inputs, outputs=net)


"""
ResNet Generator Model With Skip Connection

The architecture of Generator is a modified ResNet.

"""

def ResNet_U_Generator(norm_type='instancenorm'):
	"""Modified resnet generator model with skip-connection
    
    Args:
        norm_type: Type of normalization. Either 'batchnorm' or 'instancenorm'.
    
    Returns:
        Generator model
	"""

	inputs = tf.keras.layers.Input(shape=[224, 224, 3])
	skips = []
	
	net = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs) # (bs, 112, 112, 64)
	net = InstanceNormalization()(net)
	net = tf.keras.layers.ReLU()(net)
	# skips.append(net)

	net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net) # (bs, 56, 56, 64)
	skips.append(net)
	
	for i in range(3):
		net = residual_block_down(net, 64, norm_type=norm_type) # (bs, 56, 56, 64)

	# change dimentions
	net = residual_block_down(net, 128, strides=(2, 2), norm_type=norm_type) # (bs, 28, 28, 128)
	skips.append(net)

	for i in range(3):
		net = residual_block_down(net, 128, norm_type=norm_type) # (bs, 28, 28, 128)

	# change dimentions
	net = residual_block_down(net, 256, strides=(2, 2), norm_type=norm_type) # (bs, 14, 14, 256)
	skips.append(net)

	for i in range(5):
		net = residual_block_down(net, 256, norm_type=norm_type) # (bs, 14, 14, 256)

	# change dimentions
	net = residual_block_down(net, 512, strides=(2, 2), norm_type=norm_type) # (bs, 7, 7, 512)
	skips.append(net)

	for i in range(2):
		net = residual_block_down(net, 512, norm_type=norm_type) # (bs, 7, 7, 512)

	skips = list(reversed(skips[:]))

	for i in range(2):
		net = residual_block_up(net, 512) # (bs, 7, 7, 512)

	net = tf.keras.layers.Concatenate()([net, skips[0]])

	net = residual_block_up(net, 256, strides=(2, 2), norm_type=norm_type) # (bs, 14, 14, 256)

	for i in range(5):
		net = residual_block_up(net, 256, norm_type=norm_type) # (bs, 14, 14, 256)

	net = tf.keras.layers.Concatenate()([net, skips[1]])

	net = residual_block_up(net, 128, strides=(2, 2), norm_type=norm_type) # (bs, 28, 28, 128)

	for i in range(3):
		net = residual_block_up(net, 128, norm_type=norm_type) # (bs, 28, 28, 128)

	net = tf.keras.layers.Concatenate()([net, skips[2]])

	net = residual_block_up(net, 64, strides=(2, 2), norm_type=norm_type) # (bs, 56, 56, 64)

	for i in range(3):
		net = residual_block_up(net, 64, norm_type=norm_type) # (bs, 56, 56, 64)

	net = tf.keras.layers.Concatenate()([net, skips[3]])

	net = tf.keras.layers.UpSampling2D()(net) # (112, 112, 64)

	# net = tf.keras.layers.Concatenate()([net, skips[4]])
	
	net = tf.keras.layers.Conv2DTranspose(3, kernel_size=(4, 4), strides=(2, 2), padding='same',
	activation='sigmoid')(net) # (bs, 224, 224, 3)
	
	return tf.keras.Model(inputs=inputs, outputs=net)

"""
Discriminator

The Discriminator is a 70*70 PatchGAN.

C64-C128-C256-C512

After the last layer, a convolution is applied to map to a 1 dimensional output, followed by a Sigmoid function

Each block in the discriminator is (Conv -> Norm -> LeakyReLU(slope=0.2))

Norm is not applied to the first C64 layer.
"""
def Discriminator(norm_type='instancenorm'):
	initializer = tf.random_normal_initializer(0., 0.02)

	inp = tf.keras.layers.Input(shape=[224, 224, 3])

	x = inp

	down1 = downsample(64, 4, norm_type, False)(x) # (bs, 112, 112, 64)
	down2 = downsample(128, 4, norm_type)(down1) # (bs, 56, 56, 128)
	down3 = downsample(256, 4, norm_type)(down2) # (bs, 28, 28, 256)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (bs, 30, 30, 256)

	conv = tf.keras.layers.Conv2D(512, 4, strides=1,
								kernel_initializer=initializer,
								use_bias=False)(zero_pad1) # (bs, 27, 27, 512)
	if norm_type.lower() == 'batchnorm':
		norm1 = tf.keras.layers.BatchNormalization()(conv)
	elif norm_type.lower() == 'instancenorm':
		norm1 = InstanceNormalization()(conv)

	leaky_relu = tf.keras.layers.LeakyReLU(0.2)(norm1)
	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (bs, 29, 29, 512)

	last = tf.keras.layers.Conv2D(1, 4, strides=1,
								kernel_initializer=initializer)(zero_pad2) # (bs, 26, 26, 512)

	return tf.keras.Model(inputs=inp, outputs=last)