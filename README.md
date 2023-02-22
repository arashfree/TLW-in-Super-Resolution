# TLW-in-Super-Resolution
This is the official PyTorch implementation of ["Trainable Loss Weights in Super-Resolution"](https://arxiv.org/abs/2301.10575).

arXiv, 2023, Arash Chaichi Mellatshahi, Shohreh Kasaei

![](https://github.com/arashfree/TLW-in-Super-Resolution/blob/main/fig1.png?raw=true)

In recent years, research on super-resolution has primarily focused on the development of unsupervised models, blind networks, and the use of optimization methods in non-blind models. But, limited research has discussed the loss function in the super-resolution process. The majority of those studies have only used perceptual similarity in a conventional way. This is while the development of appropriate loss can improve the quality of other methods as well. In this article, a new weighting method for pixel-wise loss is proposed. With the help of this method, it is possible to use trainable weights based on the general structure of the image and its perceptual features while maintaining the advantages of pixel-wise loss. Also, a criterion for comparing weights of loss is introduced so that the weights can be estimated directly by a convolutional neural network using this criterion. In addition, in this article, the expectation-maximization method is used for the simultaneous estimation super-resolution network and weighting network. In addition, a new activation function, called "FixedSum", is introduced which can keep the sum of all components of vector constants while keeping the output components between zero and one. As shown in the experimental results section, weighted loss by the proposed method leads to better results than the unweighted loss in both signal-to-noise and perceptual similarity senses.


## Citation
If this article is useful for your research, please cite the article:

```
@article{ 
  mellatshahi2023trainable, 
  title={Trainable Loss Weights in Super-Resolution}, 
  author={Mellatshahi, Arash Chaichi and Kasaei, Shohreh},   
  journal={arXiv preprint arXiv:2301.10575},
  year={2023} 
}
```


