import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import cv2
import os
import time

# config

# BUFFER_SIZE = 
# BATCH_SIZE = 
# STEPS_PER_EPOCH = 
epochs = 40
learning_rate = 0.001
# LAMBDA =


work_dir = os.path.dirname(os.getcwd())

def scheduler(epoch):
	if epoch < epochs * 0.4:
		return learning_rate
	if epoch < epochs * 0.8:
		return learning_rate * 0.1
	return learning_rate * 0.01

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

def residual_block(inputs, filters, strides=(1, 1)):
	if strides == (1, 1):
		shortcut = inputs
	else:
		# decrease length and width of inputs
		shortcut = tf.keras.layers.Conv2D(filters, kernel_size=(1, 1), strides=strides)(inputs)

	# first conv2d, may decrease the size
	net = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=strides, padding='same')(inputs)
	net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	net = tf.keras.layers.ReLU()(net)

	# second conv2d, do not decrease the size
	net = tf.keras.layers.Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(net)
	net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	net = tf.keras.layers.Add()([net, shortcut])
	net = tf.keras.layers.ReLU()(net)

	return net

def ResNet(inputs):
	net = tf.keras.layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(inputs) # (bs, 112, 112, 64)
	net = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)(net)
	net = tf.keras.layers.ReLU()(net)
	net = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(net) # (bs, 56, 56, 64)

	for i in range(3):
		net = residual_block(net, 64) # (bs, 56, 56, 64)

	# change dimentions
	net = residual_block(net, 128, strides=(2, 2)) # (bs, 28, 28, 128)

	for i in range(3):
		net = residual_block(net, 128) # (bs, 28, 28, 128)

	# change dimentions
	net = residual_block(net, 256, strides=(2, 2)) # (bs, 14, 14, 256)

	for i in range(5):
		net = residual_block(net, 256) # (bs, 14, 14, 256)

	# change dimentions
	net = residual_block(net, 512, strides=(2, 2)) # (bs, 7, 7, 512)

	for i in range(2):
		net = residual_block(net, 512) # (bs, 7, 7, 512)

	net = tf.keras.layers.AveragePooling2D((2, 2), padding='same')(net) # (bs, 4, 4, 512)
	net = tf.keras.layers.Flatten()(net)
	net = tf.keras.layers.Dense(3740, activation='softmax')(net)

	return net

if __name__ == '__main__':
	# gpu config
	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	img_input = tf.keras.Input(shape=(224, 224, 3))
	output = ResNet(img_input)
	resnet = tf.keras.Model(img_input, output)

	train_datagen = ImageDataGenerator(rescale = 1./255)
	test_datagen = ImageDataGenerator(rescale = 1./255)

	train_generator = train_datagen.flow_from_directory('./ocr_data/train',
		target_size = (224, 224),
		batch_size = 32)
	test_generator = test_datagen.flow_from_directory('./ocr_data/test',
		target_size = (224, 224),
		batch_size = 32)

	change_lr = LearningRateScheduler(scheduler)
	filepath = "./ocr_data/weights_{epoch:03d}.h5"
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath, verbose=1)

	resnet.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
		loss='categorical_crossentropy',
		metrics=['accuracy'])

	resnet.fit(train_generator,
		steps_per_epoch = 19484,
		epochs = 40,
		callbacks = [change_lr, cp_callback],
		validation_data = test_generator,
		validation_steps = 1000, 
		)

