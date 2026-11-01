# Example data

`iris.data` is committed. The larger datasets are downloaded locally and are
not committed:

```bash
bash scripts/download_mnist.sh
bash scripts/download_cifar10.sh
bash scripts/download_wikipedia.sh
```

Layouts:

- `data/mnist/train-images-idx3-ubyte` and the three other IDX files
- `data/cifar10/train/<class>/*.png` and `data/cifar10/test/<class>/*.png`
  (fast.ai PNG mirror)
- `data/wikipedia.txt` (plain text extracts)
