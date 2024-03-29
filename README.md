
# Trainable Loss Weights in Super-Resolution
This is the official PyTorch implementation of ["Trainable Loss Weights in Super-Resolution"](https://arxiv.org/abs/2301.10575).

arXiv, 2023, Arash Chaichi Mellatshahi, Shohreh Kasaei

![](https://github.com/arashfree/TLW-in-Super-Resolution/blob/main/fig1.png?raw=true)

In recent years, limited research has discussed the loss function in the super-resolution process. The majority of those studies have only used perceptual similarity conventionally. This is while the development of appropriate loss can improve the quality of other methods as well. In this article, a new weighting method for pixel-wise loss is proposed. With the help of this method, it is possible to use trainable weights based on the general structure of the image and its perceptual features while maintaining the advantages of pixel-wise loss. Also, a criterion for comparing weights of loss is introduced so that the weights can be estimated directly by a convolutional neural network. In addition, in this article, the expectation-maximization method is used for the simultaneous estimation super-resolution network and weighting network. In addition, a new activation function, called “FixedSum”, is introduced which can keep the sum of all components of vector constants while keeping the output components between zero and one. As experimental results shows, weighted loss by the proposed method leads to better results than the unweighted loss and weighted loss based on uncertainty in both signal-to-noise and perceptual similarity senses on the state-of-the-art networks. Code is available online.

## To run code
### train
Run the following code to train models based on L1/MSE/L1+TLW/MSE+TLW loss.
```
python train.py --model <'RCAN'or'EDSR'or'VDSR'> --modelpath <path of folder of models> --trainpath <path of train images> --val <path of validation images> --load  --device <'cpu'or'cuda'>
```
Run the following code to train HAT model based on L1+TLW/L1+uncertainty/L1 loss.
```
python train_uncertainty.py.py --model 'HAT' --modelpath <path of folder of models> --trainpath <path of train images> --val <path of validation images> --load  --device <'cpu'or'cuda'>
```

### validation
Run the following code to validate models based on L1/MSE/L1+TLW/MSE+TLW loss on the specified dataset.
```
python val.py --model <'RCAN'or'EDSR'or'VDSR'> --modelpath <path of folder of models> --folder <path of validation images> --load --best --device <'cpu'or'cuda'>
```
Run the following code to validate HAT model based on L1+TLW/L1+uncertainty/L1 loss.
```
python val_uncertanity.py --model 'HAT' --modelpath <path of folder of models> --folder <path of validation images> --load --best --device <'cpu'or'cuda'>
```


| model| scale | url |
| --- | --- | --- |
| EDSR | x2 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| VDSR | x2 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| RCAN | x2 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| EDSR | x3 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| VDSR | x3 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| RCAN | x3 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| EDSR | x4 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| VDSR | x4 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| RCAN | x4 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |
| HAT | x4 | [download link](https://drive.google.com/drive/folders/1b6pLlMgW7UVATc6nmyknt6jCVxvbqHTU?usp=sharing) |



## Citation
If this article is useful for your research, please cite the article:

```
@article{ 
  mellatshahi2023trainable, 
  title={Trainable Loss Weights in Super-Resolution}, 
  author={Mellatshahi Chaichi, Arash and Kasaei, Shohreh},   
  journal={arXiv preprint arXiv:2301.10575},
  year={2023} 
}
```


