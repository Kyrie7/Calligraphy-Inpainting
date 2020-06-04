import tensorflow as tf
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import utils
from utils import load_image, preprocess, build_train_dataset, build_val_dataset, plot_images
from utils import generator_loss, discriminator_loss, l1_loss
import models.mmodels as mmodels

@tf.function
def train_step(mask_images, real_images, generator, discriminator, generator_optimizer, discriminator_optimizer, l1_lambda):
	with tf.GradientTape(persistent=True) as tape:
		fake_images = generator(mask_images)

		disc_real = discriminator(real_images)
		disc_fake = discriminator(fake_images)

		# cal loss
		gen_ad_loss = generator_loss(disc_fake)
		gen_l1_loss = l1_loss(real_images, fake_images, l1_lambda)
		gen_loss = gen_ad_loss + gen_l1_loss
		disc_loss = discriminator_loss(disc_real, disc_fake)

	# calculating gradients
	generator_grad = tape.gradient(gen_loss, generator.trainable_variables)
	discriminator_grad = tape.gradient(disc_loss, discriminator.trainable_variables)

	generator_optimizer.apply_gradients(zip(generator_grad, generator.trainable_variables))
	discriminator_optimizer.apply_gradients(zip(discriminator_grad, discriminator.trainable_variables))

def val_step(val_mask_bs_images, val_real_bs_images, generator, index, epoch):
	pred_images = generator(val_mask_bs_images, training=False).numpy()
	plot_images(val_mask_bs_images, val_real_bs_images, pred_images, index, epoch)


# operations on each epoch
def train(train_images, generator, discriminator, generator_optimizer, discriminator_optimizer, epoch, steps_per_epoch, batch_size, test_batch_size, l1_lambda):
	for i in tqdm(range(steps_per_epoch)):
		images_path = next(iter(train_images))
		images = []
		for image_path in images_path:
			image = load_image(image_path)
			images.append(image)
		mask_images, real_images = preprocess(images, batch_size)
		train_step(mask_images, real_images, generator, discriminator, generator_optimizer, discriminator_optimizer, l1_lambda)

def val(val_mask_images, val_real_images, generator, val_batch_size, epoch):
	for i in range(val_batch_size):
		val_mask_bs_images = val_mask_images[i*8:(i+1)*8, :, :, :]
		val_real_bs_images = val_real_images[i*8:(i+1)*8, :, :, :]
		val_step(val_mask_bs_images, val_real_bs_images, generator, i, epoch)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--batch-size', type=int, default=8, help='Batch Size')
	parser.add_argument('--val-batch-size', type=int, default=8, help='Draw Result Batch Size')
	parser.add_argument('--model-index', type=int, default=1, help='Choose Model To Train, 1 2 3 4 optional')
	parser.add_argument('--weight-decay', type=float, default=1e-4, help='Weight Decay')
	parser.add_argument('--learning-rate', type=float, default=2e-4, help='Initial Learning Rate')
	parser.add_argument('--l1-lambda', type=int, default=100, help='L1 Loss Lambda')
	parser.add_argument('--norm-type', type=str, default='instancenorm', help='Norm Type')
	parser.add_argument('--epochs', type=int, default=30, help='Epoch Nums')

	args = parser.parse_args()

	batch_size = args.batch_size
	val_batch_size = args.val_batch_size
	model_index = args.model_index
	weight_decay = args.weight_decay # to do
	learning_rate = args.learning_rate
	l1_lambda = args.l1_lambda
	norm_type = args.norm_type
	epochs = args.epochs


	physical_devices = tf.config.experimental.list_physical_devices('GPU')
	tf.config.experimental.set_memory_growth(physical_devices[0], True)

	train_images, train_nums = build_train_dataset(batch_size)
	val_mask_images, val_real_images = build_val_dataset(val_batch_size)

	steps_per_epoch = int(train_nums / batch_size)
	learning_rate = [learning_rate, learning_rate*0.1, learning_rate*0.01][:] 
	boundaries = [int(0.4 * epochs * steps_per_epoch), int(0.6 * epochs * steps_per_epoch)]

	if model_index == 1:
		generator = mmodels.Encoder_Decoder(norm_type=norm_type)
	elif model_index == 2:
		generator = mmodels.Unet_Generator(norm_type=norm_type)
	elif model_index == 3:
		generator = mmodels.ResNet_Generator(norm_type=norm_type)
	else:
		generator = mmodels.ResNet_U_Generator(norm_type=norm_type)

	discriminator = mmodels.Discriminator(norm_type=norm_type)
	generator.summary()
	discriminator.summary()

	learning_rate_schedules = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, learning_rate)
	generator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)
	discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_schedules, beta_1=0.5)

	# train
	for epoch in range(epochs):
		print('epoch: {}'.format(epoch))
		train(train_images, generator, discriminator,
			generator_optimizer, discriminator_optimizer,
			epoch, steps_per_epoch, batch_size, val_batch_size, l1_lambda)
		val(val_mask_images, val_real_images, generator, val_batch_size, epoch)



























