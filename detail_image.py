import tkinter as tk
from PIL import Image, ImageTk
import os

def show_image_detail(image_path):
    """Affiche une fenêtre avec l'image sélectionnée et son nom en titre."""
    if not os.path.exists(image_path):
        print(f"L'image {image_path} est introuvable.")
        return

    # Créer une nouvelle fenêtre
    detail_window = tk.Toplevel()
    detail_window.title(f"Détail de l'image : {os.path.basename(image_path)}")
    detail_window.geometry("800x600")

    try:
        # Charger l'image sélectionnée
        image = Image.open(image_path).resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Conserver une référence pour éviter le garbage collector
        detail_window.photo = photo

        # Ajouter le titre de l'image
        title_label = tk.Label(detail_window, text=os.path.basename(image_path), font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Ajouter l'image à la fenêtre
        image_label = tk.Label(detail_window, image=photo)
        image_label.pack(expand=True)

        # Bouton pour fermer la fenêtre
        close_button = tk.Button(detail_window, text="Fermer", command=detail_window.destroy, font=("Arial", 14))
        close_button.pack(pady=10)

    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image {image_path} : {e}")