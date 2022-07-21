import io
import os
import sys
import torch
from matplotlib import pyplot as plt
from PIL import Image

from models.resnet import ResNet
from datasets import fetch


def imagenet_categories():
    from torchvision.models import ResNet50_Weights

    return ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def preprocess(img: Image.Image) -> torch.tensor:
    import torchvision.transforms.functional as F

    img = F.resize(img, 256)
    img = F.center_crop(img, 224)
    img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img


if __name__ == "__main__":
    NUM = int(os.getenv("NUM", "18"))
    model = ResNet(num=NUM)
    model.load_from_pretrained()
    model.eval()

    url = sys.argv[1]
    if url.startswith("http"):
        img = Image.open(io.BytesIO(fetch(url)))
    else:
        img = Image.open(url)
    batch = preprocess(img).unsqueeze(0)

    # If you want to see the preprocessed image
    # plt.imshow(batch[0].permute((1, 2, 0)))

    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = imagenet_categories()[class_id]
    print(f"{category_name}: {100 * score:.1f}%")
