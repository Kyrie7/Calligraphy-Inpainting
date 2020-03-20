import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm
from copy import deepcopy

# config
BUFFER_SIZE = 3600
BATCH_SIZE = 8
STEPS_PER_EPOCH = 450
EPOCHS = 50
LAMBDA = 1000

work_dir = os.path.dirname(os.getcwd())

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
		return self.scale * normalized + self.offset

# downsample block
def downsample(filters, size, norm_type='batchnorm', apply_norm=True):
	"""Downsamples an input.
      
    Conv2D => Batchnorm => LeakyRelu
      
    Args:
        filters: number of filters
        size: filter size
        norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
        apply_norm: If True, adds the batchnorm layer
    Returns:
        Downsample Sequential Model
    """

	initializer = tf.random_normal_initializer(0., 0.02)

	block = tf.keras.Sequential()

	block.add(tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
    								kernel_initializer=initializer,
    								use_bias=False))

	if apply_norm:
		if norm_type.lower() == "batchnorm":
			block.add(tf.keras.layers.BatchNormalization())
		elif norm_type.lower() == "instancenorm":
			block.add(InstanceNormalization())

	block.add(tf.keras.layers.LeakyReLU(0.2))

	return block

# upsample block
def upsample(filters, size, norm_type='batchnorm', apply_dropout=False):
	"""Upsamples an input.
    
    Conv2DTranspose => Batchnorm => Dropout => Relu
    
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

"""
Unet - Generator

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

def Unet_Generator(norm_type='batchnorm'):
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
Discriminator

The Discriminator is a 70*70 PatchGAN.

C64-C128-C256-C512

After the last layer, a convolution is applied to map to a 1 dimensional output, followed by a Sigmoid function

Each block in the discriminator is (Conv -> Norm -> Leaky ReLU(slope=0.2))

Norm is not applied to the first C64 layer.
"""
def Discriminator(norm_type='batchnorm'):
	initializer = tf.random_normal_initializer(0., 0.02)
    
	inp = tf.keras.layers.Input(shape=[224, 224, 3])
    
	x = inp
    
	down1 = downsample(64, 4, norm_type, False)(x) # (112, 112, 64)
	down2 = downsample(128, 4, norm_type)(down1) # (56, 56, 128)
	down3 = downsample(256, 4, norm_type)(down2) # (28, 28, 256)

	zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3) # (30, 30, 256)

	conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                 kernel_initializer=initializer,
                                 use_bias=False)(zero_pad1) # (27, 27, 256)
	if norm_type.lower() == 'batchnorm':
		norm1 = tf.keras.layers.BatchNormalization()(conv)
	elif norm_type.lower() == 'instancenorm':
		norm1 = InstanceNormalization()(conv)
    
	leaky_relu = tf.keras.layers.LeakyReLU(0.2)(norm1)
	zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu) # (29, 29, 256)

	last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                 kernel_initializer=initializer)(zero_pad2) # (26, 26, 1)

	return tf.keras.Model(inputs=inp, outputs=last)

# define adversarial_loss and l1_loss
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
# the generator loss only need to fool discriminator
def generator_loss(disc_gen):
 	return loss_obj(tf.ones_like(disc_gen), disc_gen)
# the discriminator need to distinguish real and fake
def discriminator_loss(disc_real, disc_gen):
	real_loss = loss_obj(tf.ones_like(disc_real), disc_real)
	gen_loss = loss_obj(tf.zeros_like(disc_gen), disc_gen)
	return (real_loss + gen_loss) * 0.5
# L1 reconstruction loss
def l1_loss(real_image, gen_image):
	loss = tf.reduce_mean(tf.abs(real_image - gen_image))
	return loss * LAMBDA

# train_step 
@tf.function
def train_step(mask_image, real_image):
	with tf.GradientTape(persistent=True) as tape:
		fake_image = generator(mask_image)

		disc_real = discriminator(real_image)
		disc_fake = discriminator(fake_image)
        
        # 计算损失
		gen_ad_loss = generator_loss(disc_fake)
		gen_l1_loss = l1_loss(real_image, fake_image)
        # 总生成器损失
		gen_loss = gen_ad_loss + gen_l1_loss
        # 总判别器损失
		disc_loss = discriminator_loss(disc_real, disc_fake)
    # calculating gradients
	generator_grad = tape.gradient(gen_loss, generator.trainable_variables)
	discriminator_grad = tape.gradient(disc_loss, discriminator.trainable_variables)
	
	generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

# save 16 test figures
def save_figures(mask_images, real_images, model, epoch):
	predictions = model(mask_images, training=False).numpy()

	plt.figure(figsize=(24, 24))

	for i in range(16):
		plt.subplot(8, 6, 3*i+1)
		plt.title("Real Image")
		plt.imshow((real_images[i] * 255).astype(np.uint8))
		plt.axis('off')
		plt.subplot(8, 6, 3*i+2)
		plt.title("Mask Image")
		plt.imshow((mask_images[i] * 255).astype(np.uint8))
		plt.axis('off')
		plt.subplot(8, 6, 3*i+3)
		plt.title("Predicted Image")
		plt.imshow((predictions[i] * 255).astype(np.uint8))
		plt.axis('off')

	plt.savefig("./pics/image_at_epoch_{:02d}.png".format(epoch+1))


# preprocess images, including random crop to [224, 224, 3], 
# random create masks and normalize to [0, 1]
def preprocess(images, batch_size):
	real_images = []
	mask_images = []
	for idx in range(batch_size):
		image = images[idx] # [256, 256, 3]
		x = np.random.randint(33)
		y = np.random.randint(33)
		real_image = deepcopy(image[x: x+224, y: y+224, :])# crop to 224×224
		mask_image = deepcopy(image[x: x+224, y: y+224, :])
        
        # 先处理mask_on_black
		mask_path = os.path.join(work_dir, "datasets\\mask_on_black\\block_")
        # black block is from 148 pics
		select = np.random.randint(148, size=(4, 4))
        # 16 blocks integrate to a whole-mask (56×56 each block)
		mask = np.zeros((224, 224, 3))
		for i in range(4):
			for j in range(4):
				block = cv2.imread(mask_path + str(select[i][j]) + '.png')
				mask[i*56:(i+1)*56, j*56:(j+1)*56, :] = block[:, :, :]
        # white color on mask is transferred to image
		mask_image[mask == 255] = 255
        
        # mask_on_white
		mask_path = os.path.join(work_dir, "datasets\\mask_on_white\\block_")
        # black block is from 148 pics
		select = np.random.randint(12, size=3)
		for i in select:
			block = cv2.imread(mask_path + str(i) + ".png")
			x = np.random.randint(50, 156)
			y = np.random.randint(50, 156)
			mask_image[x:x+28, y:y+28, :] = block[:, :, :]

		real_images.append(real_image)
		mask_images.append(mask_image)

	real_images = np.array(real_images)
	mask_images = np.array(mask_images)

	real_images = real_images.astype(np.float32)
	mask_images = mask_images.astype(np.float32)
	real_images /= 255.
	mask_images /= 255.
	return mask_images, real_images

# operations on each epoch
def train(train_images, epoch):
    np.random.shuffle(train_images)
    
    for i in tqdm(range(STEPS_PER_EPOCH)):
        images = train_images[i * BATCH_SIZE: (i + 1) * BATCH_SIZE, :, :, :]
        mask_images, real_images = preprocess(images, BATCH_SIZE)
        train_step(mask_images, real_images)
            
    save_figures(examples_mask, examples_real, generator, epoch)

if __name__ == '__main__':
	# gpu config
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	# load data
	train_images = np.load(os.path.join(work_dir, "datasets\\xingkai\\train_images.npy"))
	test_images = np.load(os.path.join(work_dir, "datasets\\xingkai\\test_images.npy"))

	# binarize
	train_images[train_images >= 127.5] = 255
	train_images[train_images < 127.5] = 0
	test_images[test_images >= 127.5] = 255
	test_images[test_images < 127.5] = 0

	# build model
	generator = Unet_Generator(norm_type='instancenorm')
	discriminator = Discriminator(norm_type='instancenorm')
	generator.summary()
	discriminator.summary()

	# optimizer
	# We use minibatch SGD and apply the Adam solver, with learning rate 0.0002, 
	# momentum parameters β1 = 0.5, β2 = 0.999.
	generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

	# extract 16 figures from test_images
	np.random.shuffle(test_images)
	examples = test_images[:16]
	examples_mask, examples_real = preprocess(examples, 16)

	# train
	for epoch in range(EPOCHS):
		print('epoch %d', epoch)
		train(train_images, epoch)  



























