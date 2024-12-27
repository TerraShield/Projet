import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

def show_image_detail(image_path):
    """Affiche une fenêtre avec l'image sélectionnée, son nom en titre, et permet d'ajouter l'image à un dossier."""
    if not os.path.exists(image_path):
        print(f"L'image {image_path} est introuvable.")
        return

    # Créer une nouvelle fenêtre
    detail_window = tk.Toplevel()
    detail_window.title(f"Détail de l'image : {os.path.basename(image_path)}")
    detail_window.geometry("800x600")

    try:
        # Charger l'image sélectionnée
        image = Image.open(image_path).resize((400, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Conserver une référence pour éviter le garbage collector
        detail_window.photo = photo

        # Ajouter le titre de l'image
        title_label = tk.Label(detail_window, text=os.path.basename(image_path), font=("Arial", 16, "bold"))
        title_label.pack(pady=10)

        # Ajouter l'image à la fenêtre
        image_label = tk.Label(detail_window, image=photo)
        image_label.pack(expand=True)

        # Fonction pour sélectionner un dossier et copier l'image
        def add_to_folder():
            folder_selected = filedialog.askdirectory(title="Sélectionnez un dossier")
            if folder_selected:
                try:
                    destination = os.path.join(folder_selected, os.path.basename(image_path))
                    with open(image_path, 'rb') as src, open(destination, 'wb') as dst:
                        dst.write(src.read())
                    success_label = tk.Label(detail_window, text="Image ajoutée avec succès !", font=("Arial", 12), fg="green")
                    success_label.pack(pady=10)
                except Exception as e:
                    error_label = tk.Label(detail_window, text=f"Erreur : {e}", font=("Arial", 12), fg="red")
                    error_label.pack(pady=10)

        # Bouton pour ajouter l'image à un dossier
        add_button = tk.Button(detail_window, text="Ajouter au dossier", command=add_to_folder, font=("Arial", 14))
        add_button.pack(pady=10)

        # Bouton pour fermer la fenêtre
        close_button = tk.Button(detail_window, text="Fermer", command=detail_window.destroy, font=("Arial", 14))
        close_button.pack(pady=10)

    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image {image_path} : {e}")
