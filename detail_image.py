import tkinter as tk
from PIL import Image, ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def show_image_detail(image_path):
    """Affiche une fenêtre avec l'image sélectionnée, son nom en titre et une carte de chaleur des pixels."""
    if not os.path.exists(image_path):
        print(f"L'image {image_path} est introuvable.")
        return

    # Créer une nouvelle fenêtre
    detail_window = tk.Toplevel()
    detail_window.title(f"Détail de l'image : {os.path.basename(image_path)}")
    detail_window.geometry("1000x800")

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

        # Générer une carte de chaleur basée sur les pixels
        img_array = np.array(image.convert("L"))  # Convertir en niveaux de gris
        heatmap = np.abs(np.fft.fft2(img_array))  # Exemple d'analyse, ajustez si nécessaire

        # Créer une figure matplotlib
        fig, ax = plt.subplots(figsize=(6, 4))
        cax = ax.imshow(heatmap, cmap="hot", interpolation="nearest")
        fig.colorbar(cax, ax=ax, orientation='vertical')
        ax.set_title("Carte de chaleur des pixels")

        # Ajouter la figure à Tkinter
        canvas = FigureCanvasTkAgg(fig, master=detail_window)
        canvas_widget = canvas.get_tk_widget()
        canvas_widget.pack(pady=10)

        # Bouton pour fermer la fenêtre
        close_button = tk.Button(detail_window, text="Fermer", command=detail_window.destroy, font=("Arial", 14))
        close_button.pack(pady=10)

    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image {image_path} : {e}")
