import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
import subprocess
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class HeatmapPage:
    def __init__(self, frame):
        self.create_heatmap_page(frame)

    def create_heatmap_page(self, frame):
        """Crée et affiche toutes les cartes thermiques d'un dossier dans un onglet avec barre de défilement."""

        # Dossier contenant les images
        image_folder = "images"
        if not os.path.exists(image_folder):
            error_label = tk.Label(frame, text="Le dossier 'images' est introuvable.", font=("Arial", 16), fg="red")
            error_label.pack(pady=20)
            return

        # Créer un Canvas pour le défilement
        canvas = tk.Canvas(frame, width=1980, height=720)
        scroll_bar_vertical = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scroll_bar_vertical.set)

        # Créer un Frame dans le Canvas pour y placer les heatmaps
        images_frame = tk.Frame(canvas)
        canvas.create_window((0, 0), window=images_frame, anchor="nw")

        # Placer le Canvas et la barre de défilement verticale
        canvas.pack(side="left", fill="both", expand=True)
        scroll_bar_vertical.pack(side="right", fill="y")

        # Charger et afficher toutes les images
        image_files = [f for f in os.listdir(image_folder) if f.endswith((".jpg", ".png", ".jpeg"))]
        if not image_files:
            no_results_label = tk.Label(frame, text="Aucune image trouvée.", font=("Arial", 16), fg="red")
            no_results_label.pack(pady=20)
            return

        # Charger les cartes thermiques
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            canvas_tk = self.generate_heatmap(image_path)
            if canvas_tk:
                widget = canvas_tk.get_tk_widget()
                widget.pack(pady=10)  # Utiliser `pack` pour aligner les heatmaps verticalement

        # Mise à jour de la zone défilable du canvas
        canvas.update_idletasks()
        canvas.config(scrollregion=canvas.bbox("all"))

    def generate_heatmap(self, image_path):
        """Génère une carte thermique à partir d'une image."""
        try:
            # Charger l'image
            image = Image.open(image_path)
            image = image.resize((200, 200))  # Redimensionner l'image pour l'affichage
            image_array = np.array(image)

            # Créer une carte thermique
            fig, ax = plt.subplots()
            cax = ax.imshow(image_array, cmap='hot', interpolation='nearest')
            fig.colorbar(cax)

            # Intégrer la carte thermique dans Tkinter
            canvas = FigureCanvasTkAgg(fig, master=tk.Frame())
            canvas.draw()

            # Fermer la figure pour libérer la mémoire
            plt.close(fig)

            return canvas
        except Exception as e:
            print(f"Erreur lors de la génération de la carte thermique : {e}")
            return None
