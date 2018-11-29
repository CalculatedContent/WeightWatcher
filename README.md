## Weight Watcher

**Weight Watcher** analyzes weight matrices of Deep Neural Networks.

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
