# Loading

```python
from multiverse.datasets import load_dataset

X_train, y_train = load_dataset("BasicMotions", split="train")
X_test, y_test = load_dataset("BasicMotions", split="test")
```

By default, downloads are cached at `~/.multiverse/datasets`.
Set `MULTIVERSE_CACHE` to override.
