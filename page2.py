import tkinter as tk
from PIL import Image, ImageTk
import os
import subprocess

# Fonction pour ouvrir un fichier image avec l'application par défaut
def open_image_file(image_path):
    """Ouvre le fichier image avec l'application par défaut."""
    try:
        subprocess.run(["start", image_path], shell=True, check=True)  # Sur Windows
    except Exception as e:
        print(f"Erreur lors de l'ouverture du fichier : {e}")


def crop_image(image):
    """Recadre l'image pour enlever les espaces vides (blancs ou transparents) autour de l'image."""
    bbox = image.getbbox()
    if bbox:
        return image.crop(bbox)
    else:
        return image


def search_images(image_folder, query):
    """Filtrer les images qui contiennent la chaîne de recherche dans leur nom."""
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.gif'))]
    return [f for f in image_files if query.lower() in f.lower()]


def create_page2(frame):
    """Crée et affiche toutes les images d'un dossier dans un onglet avec barre de défilement."""

    # Dossier contenant les images
    image_folder = "images"
    if not os.path.exists(image_folder):
        error_label = tk.Label(frame, text="Le dossier 'images' est introuvable.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    # Créer un Frame pour la barre de recherche
    search_frame = tk.Frame(frame)
    search_frame.pack(pady=10, anchor="w")

    search_label = tk.Label(search_frame, text="Rechercher :", font=("Arial", 12))
    search_label.pack(side="left", padx=5)

    search_entry = tk.Entry(search_frame, font=("Arial", 12))
    search_entry.pack(side="left", padx=5)

    def on_search():
        query = search_entry.get()  # Récupérer la chaîne de recherche
        update_gallery(query)

    search_button = tk.Button(search_frame, text="Chercher", font=("Arial", 12), command=on_search)
    search_button.pack(side="left", padx=5)

    # Fonction pour mettre à jour la galerie avec les résultats de la recherche
    def update_gallery(query=""):
        """Met à jour la galerie en fonction du texte de recherche."""
        for widget in images_frame.winfo_children():
            widget.destroy()  # Efface les images précédemment affichées

        image_files = search_images(image_folder, query)
        
        if not image_files:
            no_results_label = tk.Label(frame, text="Aucune image trouvée.", font=("Arial", 16), fg="red")
            no_results_label.pack(pady=20)
            return

        # Créer une liste pour les objets d'images affichés
        image_labels = []

        # Taille fixe pour redimensionner toutes les images après recadrage
        fixed_width = 200
        fixed_height = 200

        # Charger et afficher toutes les images
        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = Image.open(image_path)
            image = crop_image(image)
            image = image.resize((fixed_width, fixed_height))

            photo = ImageTk.PhotoImage(image)

            label = tk.Label(images_frame, image=photo)
            label.image = photo
            image_labels.append(label)

            # Ajouter un événement de clic pour ouvrir l'image avec l'application par défaut
            label.bind("<Button-1>", lambda e, image_path=image_path: open_image_file(image_path))

        # Aligner toutes les images en grille
        row = 0
        col = 0
        max_images_per_row = 5  # Nombre d'images par ligne

        for label in image_labels:
            label.grid(row=row, column=col, padx=10, pady=10)
            col += 1
            if col >= max_images_per_row:
                col = 0
                row += 1

        # Mise à jour de la zone défilable du canvas
        canvas.config(scrollregion=canvas.bbox("all"))

    # Créer un Canvas pour le défilement
    canvas = tk.Canvas(frame, width=1980, height=720)  # Taille ajustée ici
    scroll_bar_horizontal = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    canvas.configure(xscrollcommand=scroll_bar_horizontal.set)
    scroll_bar_vertical = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    canvas.configure(yscrollcommand=scroll_bar_vertical.set)

    # Créer un Frame dans le Canvas pour y placer les images
    images_frame = tk.Frame(canvas)
    canvas.create_window((0, 0), window=images_frame, anchor="nw")

    # Placer la scrollbar en bas et à droite
    scroll_bar_horizontal.pack(fill="x", side="bottom")
    scroll_bar_vertical.pack(fill="y", side="right")
    canvas.pack(fill="both", expand=True)

    # Charger les images initiales (toutes les images sans filtre)
    update_gallery()  # Affiche toutes les images au début

    # Mettre à jour la scrollregion après avoir ajouté toutes les images
    canvas.update_idletasks()  # Actualiser les tâches en arrière-plan, nécessaire pour recalculer la zone défilable
    canvas.config(scrollregion=canvas.bbox("all"))  # Actualiser la région défilable du canvas
