import numpy as np

import sys
sys.path.append('..')
from utils.patch import reshape_patch

def get_channel_idx(total_channel, channel_using):
	return [i for i in range(total_channel) if (i%channel_using==(channel_using-1))\
											 or (i%channel_using==(channel_using-2))]
def future_replace():
	pass


if __name__ == '__main__':
	import cv2, torch
	import matplotlib.pyplot as plt

	from skimage.io import imread
	from skimage import transform

	from time import time

	imgs = []
	for i in range(5):
		img = imread('img/%d.png'%(i), 0)[:, :, 3]
		img = transform.resize(img, (320, 256))
		print(img.shape)
		imgs.append(img[:,:,None])

	futures = []
	for i in range(2):
		img = imread('img/%d.png'%(6+i), 0)[:, :, 3]
		img = transform.resize(img, (320, 256))

		print(img.shape)
		futures.append(img[:,:,None])

	imgs = np.concatenate(imgs, axis=2)[None, None]
	futures = np.concatenate(futures, axis=2)[None, None]

	imgs_patched = reshape_patch(torch.tensor(imgs), patch_size=4).numpy()
	futures_patched = reshape_patch(torch.tensor(futures), patch_size=4).numpy()
	print(imgs.shape)
	# build channel index
	real_channel = 5
	future_channel = 2


	total_channel = imgs_patched.shape[-1]
	print(total_channel)

	t = time()
	channel_idx = get_channel_idx(total_channel, imgs.shape[-1])
	# [i for i in range(total_channel) if (i%5==3) or (i%5==4)]	
	# print(channel_idx, len(channel_idx))

	imgs_patched[:, :, :, :, channel_idx] = futures_patched
	print(time()-t)

	fig, axes = plt.subplots(8,10)
	ax = axes.flatten()

	for i in range(80):
		ax[i].imshow(imgs_patched[0, 0,:, :, i])
		ax[i].set_axis_off()
		ax[i].set_title(i)
	plt.tight_layout()
	plt.show()

