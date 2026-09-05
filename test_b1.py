import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import ResNet50_Weights

from efficientvit.cls_model_zoo import create_efficientvit_cls_model


# 1. Load pretrained EfficientViT-B1
model = create_efficientvit_cls_model(
    name="efficientvit-b1-r224",
    pretrained=True,
)
model.eval()


# 2. ImageNet preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# 3. Load image
image = Image.open("Loki.jpg").convert("RGB")

x = preprocess(image)

# Add batch dimension:
# [3, 224, 224] -> [1, 3, 224, 224]
x = x.unsqueeze(0)


# 4. Inference
with torch.no_grad():
    logits = model(x)

probabilities = torch.softmax(logits, dim=1)


# 5. ImageNet class names
classes = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]


# 6. Top-5 predictions
top5_prob, top5_class = torch.topk(probabilities, 5)

for prob, class_id in zip(top5_prob[0], top5_class[0]):
    print(
        f"{class_id.item():4d}  "
        f"{classes[class_id.item()]:25s} "
        f"{prob.item() * 100:.2f}%"
    )