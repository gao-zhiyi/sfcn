import numpy as np
import torch, json
import torch.nn as nn
from PIL import Image
import onnxruntime
# from model.fbcn_lstm import FBCNLSTM


def pred_long(model, data_in, future, t_pred_long, t_pred, t_in, deep, location):
    data_list = []

    for i in range(t_pred_long//t_pred):

        data_in = np.concatenate([data_in, deep], axis=4)
        data_in = np.concatenate([data_in, location], axis=4)

        ort_inputs = {'x': data_in, 
                      'future_input': future[:, t_pred*i:t_pred*(i+1)]}
        pred = model.run(None, ort_inputs)[0]

        pred = pred[:, -t_pred:, :, :, :-2]

        data_in = np.concatenate([pred[:, -t_in:], 
                                 future[:, t_pred*(i+1)-t_in: t_pred*(i+1)]], 4)

        data_list.append(pred)

    pred = np.concatenate(data_list, axis=1)[:, :, :, :, 0]
    return pred

class SFCN(object):
    """docstring for ConvLSTM"""
    def __init__(self):
        super(SFCN, self).__init__()
        print('using onnx sfcn!')

        deep = np.asarray(Image.open('static_map/deep_india.tif'))
        deep = deep[None, :, :, None]

        location_xx = np.asarray(Image.open('static_map/xx.tif'))
        location_yy = np.asarray(Image.open('static_map/yy.tif'))
        location_xx = location_xx[None, :, :, None]
        location_yy = location_yy[None, :, :, None]

        location = np.concatenate([location_xx, location_yy], axis=3)

        self.deep = np.tile(deep, (3, 1, 1, 1))[None]

        self.location = np.tile(location, (3, 1, 1, 1))[None]

        self.model = onnxruntime.InferenceSession("model/sfcn.onnx")

    def infer(self, x, future, t_pred_long):
        with torch.no_grad():
            pred = pred_long(self.model, x, future,
                             t_pred_long, t_pred=8, t_in=3, 
                             deep=self.deep, location=self.location)*18.06
        return pred

if __name__ == '__main__':
    # (1, 3, 384, 300, 3) (1, 120, 384, 300, 2)
    x = torch.randn((1, 3, 384, 300, 3))
    wind = torch.randn((1, 120, 384, 300, 2))
    net = SFCN()
    net.infer(x, wind, 120)


