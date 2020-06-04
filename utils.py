import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import time
import pathlib
import random
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_image(image_path):
	try:
		image = tf.io.read_file(image_path)
		image = tf.io.decode_png(image, channels=3)
		return image
	except Exception as e:
		print(image_path)

# the generator loss only need to fool discriminator
def generator_loss(disc_gen):
	loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	return loss_obj(tf.ones_like(disc_gen), disc_gen)
# the discriminator need to distinguish real and fake
def discriminator_loss(disc_real, disc_gen):
	loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
	real_loss = loss_obj(tf.ones_like(disc_real), disc_real)
	gen_loss = loss_obj(tf.zeros_like(disc_gen), disc_gen)
	return (real_loss + gen_loss) * 0.5
# L1 reconstruction loss
def l1_loss(real_image, gen_image, l1_lambda):
	loss = tf.reduce_mean(tf.abs(real_image - gen_image))
	return loss * l1_lambda
"""
def save_figures(images, epoch):
	fig, axes = plt.subplots(20, 1, figsize=(20, 20))
	axes = axes.flatten()
	for img, ax in zip(images, axes):
		ax.imshow(img)
		ax.axis('off')
	plt.tight_layout()
	plt.savefig('./pics/epoch_{:02d}.png'.format(epoch+1))
"""

def preprocess(images, batch_size):
	real_images = []
	mask_images = []
	for i in range(batch_size):
		real_image = images[i].numpy()
		mask_image = images[i].numpy()
		x = np.random.randint(33)
		y = np.random.randint(33)
		real_image = real_image[x: x+224, y: y+224, :][...]
		mask_image = mask_image[x: x+224, y: y+224, :][...]
		black_mask_path = ('./data/blackmasks/block_')
		select = np.random.randint(148, size=(4, 4))
		mask = np.zeros((224, 224, 3))
		for i in range(4):
			for j in range(4):
				block = load_image(black_mask_path + str(select[i][j]) + '.png')
				block = block.numpy()
				mask[i*56:(i+1)*56, j*56:(j+1)*56, :] = block[:, :, :]
		mask_image[mask == 255] = 255
		white_mask_path = './data/whitemasks/block_'
		select = np.random.randint(12, size=3)
		for i in select:
			block = load_image(white_mask_path + str(i) + '.png')
			block = block.numpy()
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


def build_train_dataset(batch_size):
	train_path = pathlib.Path('./data/train')
	train_images_paths = list(train_path.glob('*/*'))
	train_images_paths = [str(path) for path in train_images_paths]
	for _ in range(3):
		random.shuffle(train_images_paths)
	image_count = len(train_images_paths)
	path_ds = tf.data.Dataset.from_tensor_slices(train_images_paths)
	# image_ds = path_ds.map(load_image)
	ds = path_ds.shuffle(buffer_size=image_count)
	ds = ds.repeat()
	ds = ds.batch(batch_size)
	return ds, image_count


def build_val_dataset(val_batch_size):
	val_path = pathlib.Path('./data/val')
	val_images_paths = list(val_path.glob('./*.png'))
	val_images_paths = [str(path) for path in val_images_paths]
	val_images = []
	for val_image_path in val_images_paths:
		val_image = load_image(val_image_path)
		val_images.append(val_image)
	val_mask_images, val_real_images = preprocess(val_images, val_batch_size*8)
	return val_mask_images, val_real_images

def plot_images(mask_images, real_images, pred_images, index, epoch):
	plt.figure(figsize=(24,24))
	for i in range(8):
		plt.subplot(8, 3, 3*i+1)
		plt.title('Real Image')
		plt.imshow((real_images[i] * 255).astype(np.uint8))
		plt.axis('off')
		plt.subplot(8, 3, 3*i+2)
		plt.title('Mask Image')
		plt.imshow((mask_images[i] * 255).astype(np.uint8))
		plt.axis('off')
		plt.subplot(8, 3, 3*i+3)
		plt.title('Pred Image')
		plt.imshow((pred_images[i] * 255).astype(np.uint8))
		plt.axis('off')
	plt.tight_layout()
	plt.subplots_adjust(left=0.25, bottom=None, right=0.75, top=None)
	plt.savefig('./pics/image_at_epoch_{:02d}_{:02d}.png'.format(epoch+1, index))
	plt.close()


