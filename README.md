## Weight Watcher

**Weight Watcher** analyzes the Fat Tails in the  weight matrices of Deep Neural Networks (DNNs).

This tool can predict the trends in the generalization accuracy of a series of DNNs, such as VGG11, VGG13, ...,
or even the entire series of ResNet models--without needing a test set !

This relies upon recent research into the [Heavy (Fat) Tailed Self Regularization in DNNs](https://openreview.net/forum?id=SJeFNoRcFQ)
 
The tool lets one compute a averager capacity, or quality, metric for a series of  DNNs, trained on the same data, but with different hyperparameters, or even different but related architectures. For example, it can predict that VGG19_BN generalizes better than VGG19, and better than VGG16_BN, VGG16, etc.--without needing the ImageNet datasets.


![alt text](https://github.com/CalculatedContent/PredictingTestAccuracies/blob/master/img/vgg-w_alphas.png)


### Types of Capacity Metrics:
There are 2 metrics availabe. The average **log Norm**, which is much faster but less accurate.
The average **weighted alpha** isn more accurate but much slower because it needs to both compute the SVD of the layer weight matrices, and then
fit the singluar/eigenvalues to a power law.

- log Norm (fast, less accurate)
- weighted alpaha (slow, more accurate)


### Frameworks supported

- Keras
- PyTorch


### Layers supported 

- Dense / Linear / Fully Connected (and Conv1D)
- Conv2D




## Installation

```sh
pip install weightwatcher
```

## Usage

Weight Watcher works with both Keras and pyTorch models.

```python
import weightwatcher as ww
watcher = ww.WeightWatcher(model=model)
results = watcher.analyze()

watcher.get_summary()
watcher.print_results()
```

## Links

[Weight Watcher homepage](https://calculationconsulting.com)

## License

[Apache License 2.0](LICENSE)
