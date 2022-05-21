import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('..')
import draw.color as color

# plt.switch_backend('agg')
from mpl_toolkits.basemap import Basemap

lon = np.load('draw/lon.npy')
lat = np.load('draw/lat.npy')

lon = lon[(lon>30)*(lon<=105)]
lat = lat[(lat>-66)*(lat<=30)][::-1]
print(lon.shape, lat.shape)

xx, yy = np.meshgrid(lon, lat)

def draw_img(y_img, pred_img, save_path, title, ax_idx=0):
    fig, axes = plt.subplots(1, 2)
    ax = axes.flatten()

    m = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-66, urcrnrlat=30,
            llcrnrlon=30, urcrnrlon=105,)
    m.ax = ax[ax_idx]
    m.shadedrelief(scale=0.2)
    mask = (y_img==0)
    y_img[mask] = None
    ax[ax_idx].pcolor(xx, yy, y_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    # ax[ax_idx].imshow(y_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    cs1 = ax[ax_idx].contour(xx, yy, np.squeeze(y_img), [2,3,4,6,9,14], vmin=0, vmax=14, colors='w', linewidths=1) 
    biaozhu = ax[ax_idx].clabel(cs1, inline=True, inline_spacing=0, fmt='%1.0f', fontsize=12)
    ax[ax_idx].set_aspect(1)
    ax[ax_idx].set_title('Ground truth', fontsize=12)
    ax[ax_idx].set_axis_off()

    # ax[1].pcolor(xx, yy, pred_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))

    m1 = Basemap(projection='cyl', resolution=None,
            llcrnrlat=-66, urcrnrlat=30,
            llcrnrlon=30, urcrnrlon=105,)
    m1.ax = ax[ax_idx+1]
    m1.shadedrelief(scale=0.2)

    pred_img[mask] = None
    ax[ax_idx+1].pcolor(xx, yy, pred_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    # ax[ax_idx+1].imshow(pred_img, vmin=0, vmax=15, cmap=plt.get_cmap('own'))
    cs1 = ax[ax_idx+1].contour(xx, yy, np.squeeze(pred_img), [2,3,4,6,9,14], vmin=0, vmax=14, colors='w', linewidths=1) 
    biaozhu = ax[ax_idx+1].clabel(cs1, inline=True, inline_spacing=0, fmt='%1.0f', fontsize=12)
    ax[ax_idx+1].set_aspect(1)
    ax[ax_idx+1].set_title('Prediction', fontsize=12)
    ax[ax_idx+1].set_axis_off()

    plt.suptitle(title, fontsize=18)
    plt.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, wspace=0.05)
    plt.savefig(save_path)


if __name__ == '__main__':

    pred = np.load('pred1.npy')
    y = np.load('y1.npy')

    print(pred.shape, y.shape)
    draw_img(y_img=y[0], pred_img=pred[0], save_path='test.png')

    # plt.imshow(y[0])
    # plt.show()

    # for time in range(5, 480, 6):

    #     print(time)
    # plt.show()
