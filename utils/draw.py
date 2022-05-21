import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from itertools import chain
from matplotlib.patches import Polygon
import matplotlib as mpl
import color
from skimage import io
import os
from PIL import Image
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Times New Roman'

def draw_screen_poly( lats, lons, m):
    x, y = m( lons, lats )
    xy = np.array([x, y]).T
    print(xy)
    poly = Polygon( xy, facecolor='forestgreen', alpha=0.7 )
    plt.gca().add_patch(poly)

def draw_img(y_img, pred_img, save_path, t):
    # fig = plt.figure(figsize=(8, 6), edgecolor='w')
    fig, axes = plt.subplots(1, 2)

    ax = axes.flatten()

    for i in range(2):
        m = Basemap(projection='cyl', resolution=None,
                    llcrnrlat=-66, urcrnrlat=30,
                    llcrnrlon=30, urcrnrlon=105,)
        m.ax = ax[i]
        m.shadedrelief(scale=0.2)

        # draw scale
        m.drawmeridians(meridians=[0, 20, 40, 60, 80, 100, 120, 140, 160, 180],
                          labels=[0, 0, 0, 1], color='none', fontsize=14)
        m.drawparallels(circles=[-70, -50, -30, -10, 10, 30, 50, 70, 90],
                          labels=[1, 0, 0, 0], color='none', fontsize=14)


    ax[0].pcolor(xx, yy, y_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    cs1 = ax[0].contour(xx, yy, np.squeeze(y_img), [2,3,4,6,9,14], vmin=0, vmax=14, colors='w', linewidths=1) 
    biaozhu = ax[0].clabel(cs1, inline=True, inline_spacing=0, fmt='%1.0f', fontsize=12)
    ax[0].set_aspect(1)
    ax[0].set_title('True', fontsize=14)

    ax[1].pcolor(xx, yy, pred_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    cs1 = ax[1].contour(xx, yy, np.squeeze(pred_img), [2,3,4,6,9,14], vmin=0, vmax=14, colors='w', linewidths=1) 
    biaozhu = ax[1].clabel(cs1, inline=True, inline_spacing=0, fmt='%1.0f', fontsize=12)
    ax[1].set_aspect(1)
    ax[1].set_title('Pred', fontsize=14)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.suptitle(f't = {t} (h)', fontsize=16)
    # plt.show()
    fig.set_size_inches(19.2, 10.8)
    plt.savefig(save_path, dpi=200, pad_inches=0)

    plt.clf()
    plt.close()



if __name__ == '__main__':

    root = '../results/india_baseline'

    y = np.load('%s/y.npy'%root)[:, :, :, :, 0]
    pred = np.load('%s/pred.npy'%root)[:, :, :, :, 0]
    lon = np.load('D:/reigon_code/data/india/lon.npy')
    lat = np.load('D:/reigon_code/data/india/lat.npy')

    lon = lon[(lon>30)*(lon<=105)]
    lat = lat[(lat>-66)*(lat<=30)][::-1]
    print(lon.shape, lat.shape)

    xx, yy = np.meshgrid(lon, lat)

    batch_idx = 1
    # t = 96
    t_pred_long = 120
    show_interval = 6

    y *= 18.06
    pred *= 18.06
    print(y.shape, pred.shape)

    pred *= (y!=0)*1
    y[y<1e-10] = None
    pred[pred==0] = None
    print(pred.max(), pred.min())

    if not os.path.exists('%s/imgs'%root):
        os.makedirs('%s/imgs'%root)

    # draw_img(y_img, pred_img, img_save_path, t)
    for batch_idx in range(24):
        image_list = []
        for t in range(0, t_pred_long, show_interval):
            img_save_path = '%s/imgs/%d.png'%(root, t)
            print(f'plot img: {t}/{t_pred_long}')
            draw_img(y_img=y[batch_idx, t],
                     pred_img=pred[batch_idx, t], 
                     save_path=f'{root}/imgs/{t}.png',
                     t=t)
            image_list.append(Image.fromarray(io.imread(f'{root}/imgs/{t}.png')))


        img, *imgs = image_list
        img.save(fp=f'{root}/result%d.gif'%batch_idx, format='GIF', append_images=imgs,
         save_all=True, duration=600, loop=0)
        # break

