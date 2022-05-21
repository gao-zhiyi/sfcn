import numpy as np
# import cv2
import matplotlib.pyplot as plt
# from matplotlib.image import mpimg
from skimage import io
from PIL import Image

def draw_fram(data, title_list, suptitle, save_flag):
    # plt.clf()
    # print(data_in.)
    num_img, w, h = data.shape

    fig, axes = plt.subplots(1, num_img)
    if num_img==1:
        ax = [axes]
    else:
        ax = axes.flatten()
    for i in range(num_img):
        ax[i].imshow(data[i], cmap='gray')
        ax[i].set_title(title_list[i])
        ax[i].set_axis_off()
    fig.tight_layout()
    plt.suptitle(suptitle, fontsize=24)
    if save_flag:
        plt.savefig('temp.png')     
    else:
        plt.show()

    plt.close()

def gif(data_in, gif_name, suptitle, interval, title_list):
    assert data_in.ndim==4
    num_img, seq_len, w, h = data_in.shape

    gif_list = []
    for i in range(seq_len//interval):
        # print(i*interval)
        draw_fram(data_in[:, i*interval], 
            suptitle='%s, %d'%(suptitle, i*interval), 
            title_list=title_list, 
            save_flag=True)
        im = io.imread('temp.png')
        # img = mping.imread('temp.png')
        # cv2.

        gif_list.append(im)
    img, *imgs = [Image.fromarray(gif_list[i]) for i in range(len(gif_list))]

    img.save(fp=gif_name, format='GIF', append_images=imgs,
             save_all=True, duration=500, loop=0)

if __name__ == '__main__':
    data = np.load('../../data/mnist_test_seq.npy').transpose((1,0,2,3))

    data_in = data[:2, ]
    print(data_in.shape)

    gif(data_in, gif_name='test.gif', interval=1, suptitle='title for test',
        title_list=['t%d'%i for i in range(5)])

