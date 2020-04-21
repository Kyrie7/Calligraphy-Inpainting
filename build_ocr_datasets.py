import numpy as np
import struct
from codecs import decode
from PIL import Image
import PIL
import os
import PIL.ImageOps
from io import StringIO, BytesIO

def load_gnt_file(filename):
	with open(filename, "rb") as f:
		while True:
			packed_length = f.read(4)
			if packed_length == b'':
				break
			length = struct.unpack("<I", packed_length)[0]
			raw_label = struct.unpack(">cc", f.read(2))
			width = struct.unpack("<H", f.read(2))[0]
			height = struct.unpack("<H", f.read(2))[0]
			photo_bytes = struct.unpack("{}B".format(height * width), f.read(height * width))
			label = decode(raw_label[0] + raw_label[1], encoding="gb2312")
			image = np.array(photo_bytes).reshape(height, width)
			image[image < 127.5] = 0
			image[image >= 127.5] = 255
			image = Image.fromarray(image)
			image = image.convert('RGB')
			image = image.resize((64, 64))
			image = PIL.ImageOps.invert(image)
			yield image, label

def save_pics(filename, tags='train'):
	cnt = 0
	data = load_gnt_file(filename)
	while True:
		try:
			image, label = next(data)
			if not os.path.exists("./data" + tags + "/" + label):
				os.mkdir("./data/" + tags + "/" + label)
			image.save("./data" + tags + "/" + label + "/" + str(cnt) + ".png")
			cnt += 1
		except StopIteration:
			break
	print("The number of images: ", cnt)

if __name__ == '__main__':
	train_filename = "./data/1.0train-gb1.gnt"
	test_filename = "./data/1.0test-gb1.gnt"
	save_pics(train_filename)
	save_pics(test_filename)


