import tkinter as tk
from tkinter import ttk
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

def create_page2(notebook, root):
    """Crée et affiche toutes les images d'un dossier dans des onglets avec barre de défilement."""

    # Dossier contenant les images
    image_folder = "images"
    if not os.path.exists(image_folder):
        error_label = tk.Label(notebook, text="Le dossier 'images' est introuvable.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    # Créer un Frame principal pour organiser les sections
    main_frame = tk.Frame(notebook)
    main_frame.pack(fill="both", expand=True)

    # Créer un Frame pour la barre de recherche
    search_frame = tk.Frame(main_frame)
    search_frame.pack(pady=10, anchor="w", fill="x")

    search_label = tk.Label(search_frame, text="Rechercher :", font=("Arial", 12))
    search_label.pack(side="left", padx=5)

    search_entry = tk.Entry(search_frame, font=("Arial", 12))
    search_entry.pack(side="left", padx=5, fill="x", expand=True)

    def on_search():
        query = search_entry.get()  # Récupérer la chaîne de recherche
        update_gallery(query)

    search_button = tk.Button(search_frame, text="Chercher", font=("Arial", 12), command=on_search)
    search_button.pack(side="left", padx=5)

    # Créer un Frame pour les onglets
    tab_frame = tk.Frame(main_frame)
    tab_frame.pack(fill="both", expand=True)

    # Créer un Notebook pour les onglets
    image_notebook = ttk.Notebook(tab_frame)
    image_notebook.pack(fill="both", expand=True)

    # Fonction pour mettre à jour la galerie avec les résultats de la recherche
    def update_gallery(query=""):
        """Met à jour la galerie en fonction du texte de recherche."""
        # Supprimer tous les onglets existants
        for tab in image_notebook.tabs():
            image_notebook.forget(tab)

        image_files = search_images(image_folder, query)

        if not image_files:
            no_results_label = tk.Label(image_notebook, text="Aucune image trouvée.", font=("Arial", 16), fg="red")
            no_results_label.pack(pady=20)
            return

        # Taille fixe pour redimensionner toutes les images après recadrage
        fixed_width = 200
        fixed_height = 200

        # Nombre d'images par onglet
        images_per_tab = 200

        # Créer des onglets pour chaque groupe de 200 images
        for i in range(0, len(image_files), images_per_tab):
            tab_frame = tk.Frame(image_notebook)
            image_notebook.add(tab_frame, text=f"Liste {i//images_per_tab + 1}")

            # Créer un Canvas pour le défilement
            canvas = tk.Canvas(tab_frame)
            scroll_bar_horizontal = tk.Scrollbar(tab_frame, orient="horizontal", command=canvas.xview)
            canvas.configure(xscrollcommand=scroll_bar_horizontal.set)
            scroll_bar_vertical = tk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            canvas.configure(yscrollcommand=scroll_bar_vertical.set)

            # Créer un Frame dans le Canvas pour y placer les images
            images_frame = tk.Frame(canvas)
            canvas.create_window((0, 0), window=images_frame, anchor="nw")

            # Placer la scrollbar en bas et à droite
            scroll_bar_horizontal.pack(fill="x", side="bottom")
            scroll_bar_vertical.pack(fill="y", side="right")
            canvas.pack(fill="both", expand=True)

            # Charger et afficher les images pour cet onglet
            row = 0
            col = 0
            max_images_per_row = 5  # Nombre d'images par ligne

            for image_file in image_files[i:i+images_per_tab]:
                image_path = os.path.join(image_folder, image_file)
                image = Image.open(image_path)
                image = crop_image(image)
                image = image.resize((fixed_width, fixed_height))

                photo = ImageTk.PhotoImage(image)

                label = tk.Label(images_frame, image=photo)
                label.image = photo

                # Ajouter un événement de clic pour ouvrir l'image avec l'application par défaut
                label.bind("<Button-1>", lambda e, image_path=image_path: open_image_file(image_path))

                # Aligner toutes les images en grille
                label.grid(row=row, column=col, padx=10, pady=10)
                col += 1
                if col >= max_images_per_row:
                    col = 0
                    row += 1

            # Mise à jour de la zone défilable du canvas
            canvas.update_idletasks()
            canvas.config(scrollregion=canvas.bbox("all"))

    # Charger les images initiales (toutes les images sans filtre)
    update_gallery()  # Affiche toutes les images au début

    # Lier les événements de la molette de la souris pour le défilement à la fenêtre principale
    def on_mouse_wheel(event):
        current_tab = image_notebook.select()
        canvas = image_notebook.nametowidget(current_tab).winfo_children()[0]
        canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    root.bind_all("<MouseWheel>", on_mouse_wheel)
