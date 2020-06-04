import numpy as np
import struct
from codecs import decode
from PIL import Image
import PIL
import os
import PIL.ImageOps
import json
from io import StringIO, BytesIO

def build_label_json():
	label_list = os.listdir('./ocr_data/train/')
	label_dict = dict()
	cnt = 0
	for label in label_list:
		label_dict[label] = cnt
		cnt += 1
	label_json = json.dumps(label_dict)
	with open("./ocr_data/label.json", 'w', encoding='utf-8') as file:
		json.dump(label_json, file)

def load_label_json():
	with open("./ocr_data/label.json", 'r', encoding='utf-8') as file:
		label_json = json.load(file)
	label_dict = json.loads(label_json)
	return label_dict


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
			image = image.resize((128, 128))
			image = PIL.ImageOps.invert(image)
			yield image, label

def save_pics(filenames, tags):
	for filename, tag in zip(filenames, tags):
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
		print("The number of " + tag + " images: ", cnt)

if __name__ == '__main__':
	filenames = ["./ocr_data/1.0train-gb1.gnt","./ocr_data/1.0test-gb1.gnt"]
	tags = ["train", "test"]
	save_pics(filenames, tags)


