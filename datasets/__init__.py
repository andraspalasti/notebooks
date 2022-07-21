def fetch(url: str) -> bytes:
    import requests, os, hashlib, tempfile

    fp = os.path.join(tempfile.gettempdir(), hashlib.md5(url.encode("utf-8")).hexdigest())
    if os.path.isfile(fp) and os.stat(fp).st_size > 0:
        with open(fp, "rb") as f:
            dat = f.read()
    else:
        print("fetching %s" % url)
        r = requests.get(url)
        assert r.status_code == 200
        dat = r.content
        with open(fp + ".tmp", "wb") as f:
            f.write(dat)
        os.rename(fp + ".tmp", fp)
    return dat


def fetch_mnist():
    import gzip, numpy as np

    parse = lambda file: np.frombuffer(gzip.open(file, "rb").read(), np.uint8).copy()
    X_train = parse("datasets/mnist/train-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28)).astype(np.float32)
    Y_train = parse("datasets/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = parse("datasets/mnist/t10k-images-idx3-ubyte.gz")[16:].reshape((-1, 28, 28)).astype(np.float32)
    Y_test = parse("datasets/mnist/t10k-labels-idx1-ubyte.gz")[8:]
    return X_train, Y_train, X_test, Y_test
