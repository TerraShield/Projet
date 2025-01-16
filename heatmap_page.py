import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from heatmap_algo import load_and_preprocess_images, extract_sift_features, create_bow, get_bow_histograms, dbscan_clustering, visualize_results, show_heatmap

def setup_heatmap_page(frame, selected_folder):
    """Configure une page pour afficher les images et leurs heatmaps.

    Args:
        frame (tk.Frame): Le cadre principal où les heatmaps seront affichées.
        selected_folder (str): Le dossier contenant les images à afficher.
    """
    if not os.path.exists(selected_folder):
        tk.Label(frame, text=f"Le dossier '{selected_folder}' est introuvable.", font=("Arial", 16), fg="red").pack(pady=20)
        return

    # Récupérer toutes les images par sous-dossier
    images_by_folder = {}
    for root_dir, _, files in os.walk(selected_folder):
        folder_name = os.path.relpath(root_dir, selected_folder)
        images_by_folder[folder_name] = [os.path.join(root_dir, file) for file in files if file.endswith(('.jpg', '.png', '.jpeg'))]

    if not any(images_by_folder.values()):
        tk.Label(frame, text="Aucune image disponible dans le dossier sélectionné.", font=("Arial", 16), fg="red").pack(pady=20)
        return

    def show_heatmap_wrapper(image_path):
        """Wrapper pour afficher la heatmap d'une image.

        Args:
            image_path (str): Chemin d'accès de l'image.
        """
        show_heatmap(image_path, images_by_folder)

    # Ajouter une barre de recherche
    search_frame = tk.Frame(frame)
    search_frame.pack(pady=10, fill="x")

    search_label = tk.Label(search_frame, text="Rechercher :", font=("Arial", 14))
    search_label.pack(side="left", padx=5)

    search_entry = tk.Entry(search_frame, font=("Arial", 14))
    search_entry.pack(side="left", padx=5, fill="x", expand=True)

    def filter_images():
        """Filtre les images en fonction de la requête de recherche."""
        query = search_entry.get().lower()
        filtered = {folder: [img for img in images if query in os.path.basename(img).lower()]
                    for folder, images in images_by_folder.items()}
        display_images(filtered)

    search_button = tk.Button(search_frame, text="Chercher", font=("Arial", 14), command=filter_images)
    search_button.pack(side="left", padx=5)

    # Créer un Notebook pour les sous-dossiers
    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True)

    def animate_click(label):
        """Créer une animation lors du clic sur une image.

        Args:
            label (tk.Label): Le label de l'image cliquée.
        """
        original_color = label.cget("bg")
        label.config(bg="yellow")
        label.after(200, lambda: label.config(bg=original_color))

    def display_images_for_folder(folder_name, images):
        """Affiche les images pour un dossier spécifique.

        Args:
            folder_name (str): Le nom du dossier.
            images (list): Liste des chemins d'accès aux images.
        """
        folder_frame = tk.Frame(notebook)
        notebook.add(folder_frame, text=folder_name)

        # Ajouter un canvas avec scroll pour chaque dossier
        canvas = tk.Canvas(folder_frame)
        scrollbar = tk.Scrollbar(folder_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas)

        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        row, col = 0, 0
        subfolder_count = 1
        for idx, image_file in enumerate(images):
            if idx > 0 and idx % 920 == 0:
                subfolder_count += 1
                folder_frame = tk.Frame(notebook)
                notebook.add(folder_frame, text=f"{folder_name} - Part {subfolder_count}")

                canvas = tk.Canvas(folder_frame)
                scrollbar = tk.Scrollbar(folder_frame, orient="vertical", command=canvas.yview)
                scrollable_frame = tk.Frame(canvas)

                scrollable_frame.bind(
                    "<Configure>",
                    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
                )

                canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
                canvas.configure(yscrollcommand=scrollbar.set)

                canvas.pack(side="left", fill="both", expand=True)
                scrollbar.pack(side="right", fill="y")

                row, col = 0, 0

            try:
                img = Image.open(image_file).resize((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(img)

                label = tk.Label(scrollable_frame, image=photo)
                label.image = photo  # Préserver une référence pour éviter le garbage collector
                label.grid(row=row, column=col, padx=10, pady=10)

                # Associer un clic gauche pour afficher les détails de l'image
                label.bind("<Button-1>", lambda e, path=image_file: [animate_click(label), show_heatmap_wrapper(path)])

                col += 1
                if col > 4:  # 5 images par ligne
                    col = 0
                    row += 1

            except Exception as e:
                print(f"Erreur lors du chargement de l'image {image_file}: {e}")

        # Lier la molette de la souris pour le défilement
        def on_mouse_wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mouse_wheel)

    def display_images(images_by_folder):
        """Affiche les images organisées par sous-dossier.

        Args:
            images_by_folder (dict): Dictionnaire contenant les chemins d'accès des images par dossier.
        """
        for folder_name, images in images_by_folder.items():
            if images:  # Ajouter un onglet uniquement si le dossier contient des images
                display_images_for_folder(folder_name, images)

    # Afficher toutes les images organisées par sous-dossier initialement
    display_images(images_by_folder)

    if notebook.tabs():  # Vérifie si des onglets existent
        notebook.select(0)  # S'assurer que la première page est affichée par défaut
