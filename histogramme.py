import tkinter as tk
import os
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

def setup_histogram_page(frame, selected_folder):
    """Configure un onglet pour afficher les histogrammes des images."""
    if not os.path.exists(selected_folder):
        tk.Label(frame, text=f"Le dossier '{selected_folder}' est introuvable.", font=("Arial", 16), fg="red").pack(pady=20)
        return

    image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        tk.Label(frame, text="Aucune image disponible dans le dossier sélectionné.", font=("Arial", 16), fg="red").pack(pady=20)
        return

    def show_histogram(image_path):
        image = Image.open(image_path).convert("RGB")
        plt.figure(figsize=(8, 6))
        plt.title(f"Histogramme de l'image : {os.path.basename(image_path)}")
        plt.xlabel("Valeur des pixels")
        plt.ylabel("Nombre de pixels")
        for i, color in enumerate(["red", "green", "blue"]):
            histogram = image.histogram()[i*256:(i+1)*256]
            plt.bar(range(256), histogram, color=color, alpha=0.6)
        plt.show()

    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    for image_file in image_files:
        try:
            img = Image.open(image_file).resize((100, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            frame_row = tk.Frame(scrollable_frame)
            frame_row.pack(pady=5, fill="x")

            img_label = tk.Label(frame_row, image=photo)
            img_label.image = photo  # Préserver une référence pour éviter le garbage collector
            img_label.pack(side="left", padx=5)

            button = tk.Button(frame_row, text="Afficher Histogramme", command=lambda path=image_file: show_histogram(path), font=("Arial", 12))
            button.pack(side="left", padx=5)

        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_file}: {e}")

    def on_mouse_wheel(event):
        canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    frame.bind_all("<MouseWheel>", on_mouse_wheel)
