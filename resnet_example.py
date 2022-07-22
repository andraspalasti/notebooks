#!/usr/bin/env python3
import io
import os
import sys
import torch
from PIL import Image
from torchvision.models import ResNet50_Weights

from models.resnet import ResNet
from datasets import fetch

CATEGORIES = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


def preprocess(img: Image.Image) -> torch.Tensor:
    import torchvision.transforms.functional as F

    img = F.resize(img, 256)
    img = F.center_crop(img, 224)
    img = F.pil_to_tensor(img)
    img = F.convert_image_dtype(img, torch.float)
    img = F.normalize(img, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return img


def infer(model: ResNet, img: Image.Image):
    batch = preprocess(img).unsqueeze(0)

    # If you want to see the preprocessed image
    # from matplotlib import pyplot as plt
    # plt.imshow(batch[0].permute((1, 2, 0)))
    # plt.show()

    pred = model(batch).squeeze(0).softmax(dim=0)
    class_id = pred.argmax()
    return CATEGORIES[class_id], pred[class_id]


if __name__ == "__main__":
    NUM = int(os.getenv("NUM", "18"))
    model = ResNet(num=NUM)
    model.load_from_pretrained()
    model.eval()

    url = sys.argv[1]
    if url == "webcam":
        import cv2

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        while True:
            _ = cap.grab()
            ret, frame = cap.read()
            img = Image.fromarray(frame[:, :, [2, 1, 0]])
            label, score = infer(model, img)
            print(f"{label}: {100 * score:.1f}%")

            aspect_ratio = frame.shape[1] / frame.shape[0]
            frame = cv2.resize(
                frame,
                dsize=(int(300 * max(aspect_ratio, 1.0)), int(300 * max(1.0 / aspect_ratio, 1.0))),
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
    else:
        if url.startswith("http"):
            img = Image.open(io.BytesIO(fetch(url)))
        else:
            img = Image.open(url)
        label, score = infer(model, img)
        print(f"{label}: {100 * score:.1f}%")
