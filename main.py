from time import time
import numpy as np
import os, shutil
from infer import SFCN
import argparse
from draw.my_draw import draw_img
from skimage import io


def get_test_data():
    x = np.load('data/1380/x.npy').transpose((0, 2, 3, 1))[None].astype(np.float32)
    future = np.load('data/1380/future.npy').transpose((0, 2, 3, 1))[None].astype(np.float32)
    y = np.load('data/1380/y.npy')[None].astype(np.float32)*18.07
    return x, future, y

if __name__ == '__main__':



    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default='data/1380', type=str)
    args = parser.parse_args()

    save_path = 'result/%s'%(args.data_path.split('/')[-1])
    print(args)
    print(save_path)

    # x, future, y = get_test_data()

    # t_pred_long = 120

    # net = SFCN()

    # t = time()
    # pred = net.infer(x, future, t_pred_long)
    # print('infer down, using: %.3f s'%(time()-t))

    # print('creat the gif now!')


    # result_save_root = 'result/model_test'
    # show_interval = 6

    # if not os.path.exists('%s/img'%result_save_root):
    #     # os.makedirs(result_save_root)
    #     os.makedirs('%s/img'%result_save_root)

    # image_list = []
    # for t in range(0, t_pred_long, show_interval):
    #     print(f'plot img: {t}/{t_pred_long}')

    #     draw_img(y[0, t], pred[0, t],  f'{result_save_root}/img/{t}.png', 't = %d h'%(t))
    #     image_list.append(Image.fromarray(io.imread(f'{result_save_root}/img/{t}.png')))

    # img, *imgs = image_list
    # img.save(fp=f'{result_save_root}/result.gif', format='GIF', append_images=imgs,
    #          save_all=True, duration=300, loop=0)

    # shutil.rmtree(f'{result_save_root}/img')

    # print('the gif is saved in:', f'{result_save_root}/result.gif')
