import io
import json
import os

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

NUM_CLASSES = 133
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


def model_fn(model_dir):
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pth"), map_location=DEVICE))
    return model.to(DEVICE).eval()


def input_fn(request_body, content_type):
    if content_type in ("application/x-image", "image/jpeg", "image/png"):
        image = Image.open(io.BytesIO(request_body)).convert("RGB")
        return transform(image).unsqueeze(0)
    raise ValueError(f"Unsupported content type: {content_type}")


def predict_fn(input_data, model):
    with torch.no_grad():
        probs = torch.softmax(model(input_data.to(DEVICE)), dim=1)[0]
    top_p, top_i = probs.topk(5)
    return [
        {
            "class_index": int(i),
            "probability": float(p)
        }
        for p, i in zip(top_p, top_i)
    ]


def output_fn(prediction, accept):
    return json.dumps({"predictions": prediction}), "application/json"
