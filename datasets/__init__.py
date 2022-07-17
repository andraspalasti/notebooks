def fetch_mnist():
    import gzip, numpy as np

    parse = lambda file: np.frombuffer(gzip.open(file, "rb").read(), np.uint8).copy()
    X_train = (
        parse("datasets/mnist/train-images-idx3-ubyte.gz")[16:]
        .reshape((-1, 28, 28))
        .astype(np.float32)
    )
    Y_train = parse("datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = (
        parse("datasets/mnist/t10k-images-idx3-ubyte.gz")[16:]
        .reshape((-1, 28, 28))
        .astype(np.float32)
    )
    Y_test = parse("datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test
