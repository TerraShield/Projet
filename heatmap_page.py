import tkinter as tk
from PIL import Image, ImageTk
import os
from detail_image import show_image_detail

def setup_heatmap_page(frame, image_folder="images"):
    """Configure l'interface utilisateur pour afficher les heatmaps."""
    image_files = []

    # Charger les fichiers images
    if os.path.exists(image_folder):
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    else:
        print(f"Dossier introuvable : {image_folder}")

    if not image_files:
        error_label = tk.Label(frame, text="Aucune image trouvée dans le dossier.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    current_index = [0]  # Utilisation d'une liste pour permettre une mutabilité

    # Frame principale
    control_frame = tk.Frame(frame)
    control_frame.pack(fill="x", pady=10)

    # Bouton pour aller à l'image précédente
    def previous_image():
        if current_index[0] > 0:
            current_index[0] -= 1
            update_image(current_index)

    # Bouton pour aller à l'image suivante
    def next_image():
        if current_index[0] < len(image_files):
            current_index[0] += 1
            update_image(current_index)

    prev_button = tk.Button(control_frame, text="\u2190", font=("Arial", 16), command=previous_image)
    prev_button.pack(side="left", padx=10)

    # Barre de recherche entre les flèches
    search_frame = tk.Frame(control_frame)
    search_frame.pack(side="left", fill="x", expand=True, padx=5)

    search_label = tk.Label(search_frame, text="Rechercher une image :", font=("Arial", 14))
    search_label.pack(side="left", padx=5)

    search_entry = tk.Entry(search_frame, font=("Arial", 14))
    search_entry.pack(side="left", fill="x", expand=True, padx=5)

    result_frame = tk.Frame(frame)
    result_frame.pack(fill="x", side="bottom", pady=10)

    result_canvas = tk.Canvas(result_frame, height=200)
    result_scrollbar = tk.Scrollbar(result_frame, orient="vertical", command=result_canvas.yview)
    result_scrollable_frame = tk.Frame(result_canvas)

    result_scrollable_frame.bind(
        "<Configure>", lambda e: result_canvas.configure(scrollregion=result_canvas.bbox("all"))
    )

    result_canvas.create_window((0, 0), window=result_scrollable_frame, anchor="nw")
    result_canvas.configure(yscrollcommand=result_scrollbar.set)

    result_canvas.pack(side="left", fill="both", expand=True)
    result_scrollbar.pack(side="right", fill="y")

    def search_image():
        for widget in result_scrollable_frame.winfo_children():
            widget.destroy()

        query = search_entry.get().strip().lower()
        matching_files = [f for f in image_files if query in f.lower()]

        if not matching_files:
            no_results_label = tk.Label(result_scrollable_frame, text="Aucun résultat trouvé.", font=("Arial", 14), fg="red")
            no_results_label.pack(pady=10)
        else:
            for file in matching_files:
                image_path = os.path.join(image_folder, file)
                try:
                    image = Image.open(image_path).resize((50, 50), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(image)

                    frame = tk.Frame(result_scrollable_frame)
                    frame.pack(fill="x", pady=5, padx=5)

                    img_label = tk.Label(frame, image=photo)
                    img_label.image = photo
                    img_label.pack(side="left", padx=5)

                    button = tk.Button(frame, text=file, font=("Arial", 12), command=lambda f=file: select_image(f))
                    button.pack(side="left", anchor="w", padx=5)

                except Exception as e:
                    print(f"Erreur lors du chargement de la vignette {image_path} : {e}")

    def select_image(file_name):
        index = image_files.index(file_name)
        current_index[0] = index
        update_image(current_index)

    search_button = tk.Button(search_frame, text="Rechercher", font=("Arial", 14), command=search_image)
    search_button.pack(side="left", padx=5)

    next_button = tk.Button(control_frame, text="\u2192", font=("Arial", 16), command=next_image)
    next_button.pack(side="right", padx=10)

    canvas = tk.Canvas(frame, width=800, height=400, bg="white")
    canvas.pack(side="top", fill="both", expand=True)

    images_cache = []  # Pour stocker les références des images chargées

    # Fonction pour afficher un ensemble d'images centré autour de l'image principale
    def update_image(index):
        canvas.delete("all")  # Effacer les images précédentes
        images_cache.clear()  # Nettoyer les références pour éviter les problèmes de mémoire
        start = max(0, index[0] - 3)
        end = min(len(image_files), index[0] + 4)
        x_offset = 400 - ((end - start) * 100) // 2  # Calcul dynamique pour centrer la séquence d'images

        for i in range(start, end):
            image_path = os.path.join(image_folder, image_files[i])
            try:
                image = Image.open(image_path).resize((150, 150), Image.Resampling.LANCZOS)
                photo = ImageTk.PhotoImage(image)
                images_cache.append(photo)  # Conserver une référence pour éviter le garbage collector

                # Dessiner l'image
                image_id = canvas.create_image(x_offset, 200, image=photo, anchor="center")

                # Ajouter un rectangle autour de l'image principale
                if i == index[0]:
                    canvas.create_rectangle(x_offset - 75, 125, x_offset + 75, 275, outline="red", width=2)

                # Associer un clic gauche pour ouvrir les détails de l'image
                canvas.tag_bind(image_id, "<Button-1>", lambda event, path=image_path: show_image_detail(path))

                x_offset += 200  # Décalage pour la prochaine image

            except Exception as e:
                print(f"Erreur lors du chargement de l'image {image_path} : {e}")

    # Initialiser avec la première image
    if image_files:
        update_image(current_index)
