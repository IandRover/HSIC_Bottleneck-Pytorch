# HSIC_Bottleneck-Pytorch Implementation

## Description

This is a pytorch implementation of HSIC bottleneck with performance of 97~98% on MNIST using similar setting as in the original paper.
This implementation is easier to understand than the [original work](https://github.com/choasma/HSIC-bottleneck). 

Compared to [this Pytorch implementation](https://github.com/gusye1234/Pytorch-HSIC-bottleneck) (I thank the author pretty much !!), our implementation works on MNIST and we did the gradient in the right way: the HSIC bottleneck does not use back propagation, so we can only perform gradient descent for each layer's weights. So stopping gradient as ```layer(input.detach())``` is necessary.

For research interests, 
1. I wonder the difference in applying student-t distribution as kernel, so one can use  ```--kernel_x "rbf" --kernel_y "rbf"``` to specify the kernel.
2. I also curious about if dropping the HSIC(X,Z_i) term or substituting it with HSIC(Z_{i-1},Z_i) yields non-inferior performance. Excitingly, the answer seems affirmative (for now XD).

### Dependencies

* torch 1.9.0
* torchvision 0.10.0

### Executing program

```
. train1.txt
```

or 
```
python v3_train.py --loss "CE" --forward "x" --kernel_x "rbf" --kernel_y "rbf"
```
## Authors
Chia-Hsiang Kao

## Version History

* 0.2
    * Various bug fixes and optimizations
    * See [commit change]() or See [release history]()
* 0.1
    * Initial Release

## License

This project is licensed under the [MIT] License

## Acknowledgments

Inspiration, code snippets, etc.
* [original work](https://github.com/choasma/HSIC-bottleneck)
* [pytorch implementation](https://github.com/gusye1234/Pytorch-HSIC-bottleneck)
