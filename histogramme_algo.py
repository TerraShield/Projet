import os
from PIL import Image
import matplotlib.pyplot as plt

def show_histogram(image_path):
    image = Image.open(image_path).convert("RGB")
    plt.figure(figsize=(8, 6))
    plt.title(f"Histogramme de l'image : {os.path.basename(image_path)}")
    plt.xlabel("Valeur des pixels")
    plt.ylabel("Nombre de pixels")
    for i, color in enumerate(["red", "green", "blue"]):
        histogram = image.histogram()[i*256:(i+1)*256]
        plt.bar(range(256), histogram, color=color, alpha=0.6)
    plt.show()
