<meta name="image" property="og:image" content="https://github.com/CalculatedContent/PredictingTestAccuracies/blob/master/img/vgg-w_alphas.png">

## Weight Watcher  

### Current Version: 0.4

**Weight Watcher** analyzes the Fat Tails in the  weight matrices of Deep Neural Networks (DNNs).

This tool can predict the trends in the generalization accuracy of a series of DNNs, such as VGG11, VGG13, ...,
or even the entire series of ResNet models--without needing a test set !

This relies upon recent research into the [Heavy (Fat) Tailed Self Regularization in DNNs](https://openreview.net/forum?id=SJeFNoRcFQ)
 
The tool lets one compute a averager capacity, or quality, metric for a series of  DNNs, trained on the same data, but with different hyperparameters, or even different but related architectures. For example, it can predict that VGG19_BN generalizes better than VGG19, and better than VGG16_BN, VGG16, etc.  



### Types of Capacity Metrics:
There are 2 basic types metrics we use

- alpha (the average power law exponent)
- weighted alpha / log_alpha_norm (scale adjusted alpha metrics)

The average **alpha**  can be used to compare one or more DNN models with different hyperparemeter settings, but of the same depth. The average **weighted alpha** is suitable for DNNs of differing depths.


Here is an example of the **Weighted Alpha** capacity metric for all the current pretrained VGG models.
![alt text](https://github.com/CalculatedContent/PredictingTestAccuracies/blob/master/img/vgg-w_alphas.png)

Notice: we *did not peek* at the ImageNet test data to build this plot.

### Frameworks supported

- Tensorflow 2.x / Keras
- PyTorch
- HuggingFace 

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
details = watcher.analyze()
summary = watcher.get_summary(details)
```

## Advanced Usage 

The analyze function has several features described below

```python
def analyze(self, model=None, layers=[], min_evals=0, max_evals=0,
                plot=True, randomize=True, mp_fit=True):
...
```

and in the [Demo Notebook](https://github.com/CalculatedContent/WeightWatcher/blob/master/WeightWatcher.ipynb)


### weighted alpha (SLOW)
Power Law fit, here with pyTorch example

```python
import weightwatcher as ww
import torchvision.models as models

model = models.vgg19_bn(pretrained=True)
watcher = ww.WeightWatcher(model=model)
details = watcher.analyze()
summary = watcher.get_summary(details)
```

It is as easy to run and generates a pandas dataframe with details (and plots) for each layer
and summary dict of generalization metrics

```python
    {'log_norm': 2.11,
      'alpha': 3.06,
      'alpha_weighted': 2.78,
      'log_alpha_norm': 3.21,
      'log_spectral_norm': 0.89,
      'stable_rank': 20.90,
      'mp_softrank': 0.52}]
```

Options include:

#### filter by layer types 

```python
results = watcher.analyze(layers=[ww.LAYER_TYPE.CONV2D])

```

#### filter by ids or name

```python
results = watcher.analyze(layers=[20])
```

#### minimum, maximum nuymber of eigenvalues  of the layer weight matrix

Sets the minimum and maximum size of the weight matrices analyzed.
Setting max is useful for a quick debugging.

```python
details = watcher.analyze(min_evals=50, max_evals=500)
```

#### plots (for weight_alpha=True)

Create log-log plots for each layer weight matrix to observe how well
the power law fits work

```python
details = watcher.analyze(plot=True)
```

[Demo Notebook](https://github.com/CalculatedContent/WeightWatcher/blob/master/WeightWatcher.ipynb)

[Calculation Consulting homepage](https://calculationconsulting.com)

[Calculated Content Blog](https://calculatedcontent.com)


---
This tool is based on state-of-the-art research done in collaboration with UC Berkeley:


- [Implicit Self-Regularization in Deep Neural Networks: Evidence from Random Matrix Theory and Implications for Learning](https://arxiv.org/abs/1810.01075)

- [Traditional and Heavy Tailed Self Regularization in Neural Network Models](https://arxiv.org/abs/1901.08276)

  - Notebook for above 2 papers (https://github.com/CalculatedContent/ImplicitSelfRegularization)

- [ICML 2019 Theoretical Physics Workshop Paper](https://github.com/CalculatedContent/PredictingTestAccuracies/blob/master/ICMLPhysicsWorkshop/icml_prl_TPDLW2019_fin.pdf)

- [Heavy-Tailed Universality Predicts Trends in Test Accuracies for Very Large Pre-Trained Deep Neural Networks](https://arxiv.org/abs/1901.08278)

  - Notebook for paper (https://github.com/CalculatedContent/PredictingTestAccuracies)


---
and has been presented at Stanford, UC Berkeley, etc:

- [NERSC Summer 2018](https://www.youtube.com/watch?v=_Ni5UDrVwYU)
- [UC Berkeley/ICSI 12/13/2018](https://www.youtube.com/watch?v=6Zgul4oygMc)

- [Institute for Pure & Applied Mathematics (IPAM)](https://www.youtube.com/watch?v=fmVuNRKsQa8)
- [Physics Informed Machine Learning](https://www.youtube.com/watch?v=eXhwLtjtUsI)

---
and major AI  conferences like ICML, KDD, etc.

- [KDD 2019 Workshop: Statistical Mechanics Methods for Discovering Knowledge from Production-Scale Neural Networks](https://dl.acm.org/doi/abs/10.1145/3292500.3332294)   

- [KDD 2019 Workshop: Slides](https://www.stat.berkeley.edu/~mmahoney/talks/dnn_kdd19_fin.pdf)
---

and has been the subject  many popular podcasts

- [This Week in ML](https://twimlai.com/meetups/implicit-self-regularization-in-deep-neural-networks/)

- [Data Science at Home Podcast](https://podcast.datascienceathome.com/e/episode-70-validate-neural-networks-without-data-with-dr-charles-martin/)

- [Aggregate Intellect Podcast](https://aisc.ai.science/events/2019-11-06)


---

## Release

Publishing to the PyPI repository:

```sh
# 1. Check in the latest code with the correct revision number (__version__ in __init__.py)
vi weightwatcher/__init__.py # Increse release number, remove -dev to revision number
git commit
# 2. Check out latest version from the repo in a fresh directory
cd ~/temp/
git clone https://github.com/CalculatedContent/WeightWatcher
cd WeightWatcher/
# 3. Use the latest version of the tools
python -m pip install --upgrade setuptools wheel twine
# 4. Create the package
python setup.py sdist bdist_wheel
# 5. Test the package
twine check dist/*
# 6. Upload the package to PyPI
twine upload dist/*
# 7. Tag/Release in github by creating a new release (https://github.com/CalculatedContent/WeightWatcher/releases/new)
```

## License

[Apache License 2.0](LICENSE.txt)

#### Contributors

[Charles H Martin, PhD](https://www.linkedin.com/in/charlesmartin14)
[Calculation Consulting](https://calculationconsulting.com)

[Serena Peng](https://www.linkedin.com/in/serenapeng)
