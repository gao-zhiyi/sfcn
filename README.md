# SFCN: State-force coupled deep spatiotemporal predictive learning for ocean wave height forecast



![](https://github.com/Prevalenter/sfcn/blob/main/imgs/result.gif)



### Dependencies

This tutorial depends on the following libraries:

* onnxruntime
* scikit-image
* pillow
* Matplotlib
* Numpy

Also, this code should be compatible with Python versions 3.8. Our code doesn't require the computer to have a GPU, nor even have pytorch installed.



### Get Started

The trained model is already included in the repository and can be quickly run with the following command

```
python main.py --data_path data/1380

python main.py --data_path data/1680

python main.py --data_path data/1740
```

the result gif will be save in result/.



the dataset used can be find here: wave and [wind](https://rda.ucar.edu/datasets/ds094.2/#!access).
