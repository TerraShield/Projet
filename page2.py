import tkinter as tk
from PIL import Image, ImageTk
import os

def create_page2(frame):
    """Crée et affiche toutes les images d'un dossier dans un onglet."""
    
    # Dossier contenant les images
    image_folder = "Images"
    
    # Vérifier si le dossier existe
    if not os.path.exists(image_folder):
        error_label = tk.Label(frame, text="Le dossier 'images' est introuvable.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return
    
    # Lister tous les fichiers dans le dossier images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
    
    if not image_files:
        error_label = tk.Label(frame, text="Aucune image trouvée dans le dossier 'images'.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return
    
    # Créer une liste pour les objets d'images affichés
    image_labels = []
    
    for image_file in image_files:
        # Charger chaque image
        image_path = os.path.join(image_folder, image_file)
        image = Image.open(image_path)
        
        # Redimensionner l'image si nécessaire (par exemple, 200x200px)
        image = image.resize((200, 200))
        
        # Convertir l'image en un format utilisable par Tkinter
        photo = ImageTk.PhotoImage(image)
        
        # Créer un label pour chaque image
        label = tk.Label(frame, image=photo)
        label.image = photo  # Garder une référence pour éviter que l'image soit détruite
        image_labels.append(label)

    # Aligner toutes les images côte à côte en utilisant pack
    for label in image_labels:
        label.pack(side="left", padx=10, pady=10)
