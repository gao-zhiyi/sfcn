import torch

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    # print(img_tensor.shape)
    batch_size, seq_length, img_height, img_width, num_channels = img_tensor.shape

    a = torch.reshape(img_tensor, [batch_size, seq_length,
                                img_height//patch_size, patch_size,
                                img_width//patch_size, patch_size,
                                num_channels])
    # b = torch.permute(a, [0,1,2,4,3,5,6])
    b = a.permute([0,1,2,4,3,5,6])
    patch_tensor = torch.reshape(b, [batch_size, seq_length,
                                  img_height//patch_size,
                                  img_width//patch_size,
                                  patch_size*patch_size*num_channels])
    return patch_tensor

def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim

    batch_size, seq_length, patch_height, patch_width, channels = patch_tensor.shape

    img_channels = channels // (patch_size*patch_size)
    a = torch.reshape(patch_tensor, [batch_size, seq_length,
                                  patch_height, patch_width,
                                  patch_size, patch_size,
                                  img_channels])
    # b = torch.transpose(a, [0,1,2,4,3,5,6])
    b = a.permute([0,1,2,4,3,5,6])
    img_tensor = torch.reshape(b, [batch_size, seq_length,
                                patch_height * patch_size,
                                patch_width * patch_size,
                                img_channels])
    return img_tensor



if __name__ == '__main__':
  import cv2
  import matplotlib.pyplot as plt
  import numpy as np
  img = cv2.imread('cat.jpeg', 1)
  print(img.shape)
  # plt.imshow(img)
  # plt.show()

  '''
  img_tensor = torch.tensor(img[None, None, :, :])
  print(img_tensor.shape)
  img_patch = reshape_patch(img_tensor, 2)
  print(img_patch.shape)
  plt.imshow(img_patch[0,0,:,:,0])
  plt.show()
  '''

  data = np.zeros((1, 1, 64, 64, 3))
  force = np.zeros((1, 1, 64, 64, 2))

  for i in range(3):
    data[:, :, :, :, i] += i

  for i in range(2):
    force[:, :, :, :, i] += (i+3)

  print([data[:, :, :, :, i].mean() for i in range(3)])
  print([force[:, :, :, :, i].mean() for i in range(2)])


  data_patched = reshape_patch(torch.tensor(data), 2).permute([0,1,4,2,3])
  force_patched = reshape_patch(torch.tensor(force), 2).permute([0,1,4,2,3])

  print([data_patched[:, :, i].mean().item() for i in range(12)])
  print([force_patched[:, :, i].mean().item() for i in range(8)])

  data_patched[:, :, ::]

  change_idx = [1, 2, 4, 5, 7, 8, 10, 11]


  print([data_patched[:, :, i].mean().item() for i in range(12)])

  data_patched[:, :, change_idx] = force_patched

  
  print([data_patched[:, :, i].mean().item() for i in range(12)])


  print(data_patched.shape,
        force_patched.shape)


