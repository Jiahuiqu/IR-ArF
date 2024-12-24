## IR&ArF: Towards Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images




The python code implementation of the paper "IR&ArF: Towards Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images"

## Requirements
    Ubuntu 22.04 cuda 11.8
    Python 3.10 Pytorch 2.4.0
    
To install requirements:
        pip install -r requirements.txt

## How to use:
### brief description
The train.py contains training code and some parameters, including the dataset path, the number of training epochs, learning rate, and so on. For more details, please refer to train.py.

The model.py include model structure.

The dataloader.py include the code for loading data.

### dataset
    the default dataset structure can be change in dataloader.py
          ─dataset_name
            ├─train
            │  ├─gtHS
            │  │  ├─1.mat
            │  │  ├─2.mat
            │  │  └─3.mat
            │  ├─hrMS
            │  ├─LRHS
            │  └─LRHS_**
            └─test

### training
    To do training with model for 200 epochs on pavia with Homography distortions, run:

```
python train.py --data /dataset/pavia \
  --dataset pavia --type _Homography2\
  --lr 0.0002 -p 200 --epochs 200
```


### testing
    To evaluate the performance on the test set, run:
    
```
python test.py 
```

### Cite
        If you find our work helpful in your research, please consider citing our paper
        

