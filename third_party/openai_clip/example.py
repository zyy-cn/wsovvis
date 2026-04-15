import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
# model, preprocess = clip.load("ViT-B/32", device=device)
# model, preprocess = clip.load("ViT-L/14", device=device)
model, preprocess = clip.load("ViT-L/14@336px", device=device)

image = preprocess(Image.open("E:/Data/PASCAL/VOCdevkit/VOC2012/JPEGImages/2007_000063.jpg")).unsqueeze(0).to(device)

# text = clip.tokenize([
#     "person, yellow",
#     "person, black",
#     "person, green",
#     "person, blue",
#     "monkey, blue",
# ]).to(device)


text = clip.tokenize([
    "black keyboard",
    "white keyboard",
    "white keyboard",
]).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)
