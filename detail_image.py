import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os

def show_image_detail(image_path):
    """Affiche une fenêtre avec l'image sélectionnée, son nom en titre, et permet d'ajouter l'image à un dossier.

    Args:
        image_path (str): Chemin d'accès de l'image à afficher.
    """
    if not os.path.exists(image_path):
        print(f"L'image {image_path} est introuvable.")
        return

    # Créer une nouvelle fenêtre
    detail_window = tk.Toplevel()
    detail_window.title(f"Détail de l'image : {os.path.basename(image_path)}")
    detail_window.geometry("800x800")

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

        def add_to_folder():
            """Ajoute l'image sélectionnée à un dossier choisi par l'utilisateur."""
            folder_selected = filedialog.askdirectory(title="Sélectionnez un dossier")
            if folder_selected:
                try:
                    destination = os.path.join(folder_selected, os.path.basename(image_path))
                    with open(image_path, 'rb') as src, open(destination, 'wb') as dst:
                        dst.write(src.read())
                    tk.Label(detail_window, text="Image ajoutée avec succès !", font=("Arial", 12), fg="green").pack(pady=10)
                except Exception as e:
                    tk.Label(detail_window, text=f"Erreur : {e}", font=("Arial", 12), fg="red").pack(pady=10)

        def rename_image():
            """Ouvre une fenêtre pour renommer l'image sélectionnée."""
            rename_window = tk.Toplevel(detail_window)
            rename_window.title("Renommer l'image")
            rename_window.geometry("300x150")

            tk.Label(rename_window, text="Entrez le nouveau nom de l'image :", font=("Arial", 12)).pack(pady=5)
            frame = tk.Frame(rename_window)
            frame.pack(pady=5)
            new_name_var = tk.StringVar(value=os.path.splitext(os.path.basename(image_path))[0])
            new_name_entry = tk.Entry(frame, textvariable=new_name_var, font=("Arial", 12))
            new_name_entry.pack(side="left")
            tk.Label(frame, text=".jpg", font=("Arial", 12)).pack(side="left")

            def apply_rename():
                """Applique le nouveau nom à l'image."""
                new_name = new_name_var.get()
                if new_name:
                    new_path = os.path.join(os.path.dirname(image_path), new_name + ".jpg")
                    try:
                        os.rename(image_path, new_path)
                        detail_window.title(f"Détail de l'image : {os.path.basename(new_path)}")
                        title_label.config(text=os.path.basename(new_path))
                        tk.Label(detail_window, text="Image renommée avec succès !", font=("Arial", 12), fg="green").pack(pady=10)
                        rename_window.destroy()
                    except Exception as e:
                        tk.Label(detail_window, text=f"Erreur : {e}", font=("Arial", 12), fg="red").pack(pady=10)

            tk.Button(rename_window, text="Appliquer", command=apply_rename, font=("Arial", 12)).pack(pady=10)

        # Bouton pour ajouter l'image à un dossier
        tk.Button(detail_window, text="Ajouter au dossier", command=add_to_folder, font=("Arial", 14)).pack(pady=10)

        # Bouton pour renommer l'image
        tk.Button(detail_window, text="Renommer l'image", command=rename_image, font=("Arial", 14)).pack(pady=10)

        # Bouton pour fermer la fenêtre
        tk.Button(detail_window, text="Fermer", command=detail_window.destroy, font=("Arial", 14)).pack(pady=10)

    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image {image_path} : {e}")
