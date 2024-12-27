import tkinter as tk
from PIL import Image, ImageTk
import os

def show_image_detail(image_path):
    """Affiche une fenêtre avec uniquement l'image sélectionnée."""
    if not os.path.exists(image_path):
        print(f"L'image {image_path} est introuvable.")
        return

    # Créer une nouvelle fenêtre
    detail_window = tk.Toplevel()
    detail_window.title("Détail de l'image")
    detail_window.geometry("800x600")

    # Charger l'image sélectionnée
    try:
        image = Image.open(image_path).resize((600, 400), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)

        # Afficher l'image dans la fenêtre
        image_label = tk.Label(detail_window, image=photo)
        image_label.image = photo  # Conserver une référence pour éviter le garbage collector
        image_label.pack(expand=True)

    except Exception as e:
        print(f"Erreur lors de l'affichage de l'image {image_path} : {e}")
        return

    # Ajouter un bouton pour fermer la fenêtre
    close_button = tk.Button(detail_window, text="Fermer", command=detail_window.destroy, font=("Arial", 14))
    close_button.pack(pady=10)

def bind_to_detail_page(frame, image_folder, image_files):
    """Ajoute des interactions pour rediriger vers la page de détails depuis la page heatmap."""
    for widget in frame.winfo_children():
        widget.destroy()

    for index, image_file in enumerate(image_files):
        try:
            image_path = os.path.join(image_folder, image_file)

            small_img = Image.open(image_path).resize((80, 60), Image.Resampling.LANCZOS)
            tk_img = ImageTk.PhotoImage(small_img)

            btn = tk.Button(frame, image=tk_img, compound="center",
                            command=lambda path=image_path: show_image_detail(path))
            btn.image = tk_img  # Conserver une référence pour éviter le garbage collector
            btn.grid(row=index // 5, column=index % 5, padx=10, pady=10)

        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_file} : {e}")
