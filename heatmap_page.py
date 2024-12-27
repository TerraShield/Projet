import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os
from detail_image import show_image_detail

def setup_heatmap_page(frame, selected_folder):
    """Configure une page de heatmap avec des sous-onglets pour chaque sous-dossier, navigation horizontale, une barre de recherche avec défilement et molette."""
    image_folder = selected_folder if selected_folder else "images"

    # Récupérer les images par sous-dossier
    images_by_folder = {}
    for root_dir, _, files in os.walk(image_folder):
        folder_name = os.path.relpath(root_dir, image_folder)
        images_by_folder[folder_name] = []
        for file in files:
            if file.endswith(('.jpg', '.png', '.jpeg')):
                images_by_folder[folder_name].append(os.path.join(root_dir, file))

    if not any(images_by_folder.values()):
        error_label = tk.Label(frame, text="Aucune image trouvée dans les sous-dossiers.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    # Créer une barre de recherche avec des flèches
    search_frame = tk.Frame(frame)
    search_frame.pack(pady=10, fill="x")

    prev_button = tk.Button(search_frame, text="\u2190", font=("Arial", 14))
    prev_button.pack(side="left", padx=5)

    search_label = tk.Label(search_frame, text="Rechercher :", font=("Arial", 14))
    search_label.pack(side="left", padx=5)

    search_entry = tk.Entry(search_frame, font=("Arial", 14))
    search_entry.pack(side="left", padx=5, fill="x", expand=True)

    next_button = tk.Button(search_frame, text="\u2192", font=("Arial", 14))
    next_button.pack(side="right", padx=5)

    notebook = ttk.Notebook(frame)
    notebook.pack(fill="both", expand=True)

    # Fonction pour afficher les images d'un sous-dossier
    def display_images_for_folder(folder_name, images):
        folder_frame = tk.Frame(notebook)
        notebook.add(folder_frame, text=folder_name)

        canvas = tk.Canvas(folder_frame, width=800, height=400, bg="white")
        canvas.pack(fill="both", expand=True)

        images_cache = []
        current_index = [0]

        def update_images_around(index):
            canvas.delete("all")
            images_cache.clear()

            start_index = max(0, index[0] - 4)
            end_index = min(len(images), index[0] + 5)
            images_to_display = images[start_index:end_index]

            for i, image_path in enumerate(images_to_display):
                try:
                    img = Image.open(image_path).resize((200, 200), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)
                    images_cache.append(photo)

                    x_position = 200 * (i - (index[0] - start_index)) + 100
                    canvas.create_image(x_position, 200, image=photo, anchor="center")

                    # Ajouter un rectangle rouge autour de l'image active
                    if i == (index[0] - start_index):
                        canvas.create_rectangle(x_position - 100, 100, x_position + 100, 300, outline="red", width=2)

                    # Associer un clic pour ouvrir les détails
                    canvas.tag_bind("current", "<Button-1>", lambda e, path=image_path: show_image_detail(path))

                except Exception as e:
                    print(f"Erreur lors du chargement de l'image {images[index[0]]}: {e}")

        def previous_image():
            if current_index[0] > 0:
                current_index[0] -= 1
                update_images_around(current_index)

        def next_image():
            if current_index[0] < len(images) - 1:
                current_index[0] += 1
                update_images_around(current_index)

        prev_button.config(command=previous_image)
        next_button.config(command=next_image)

        if images:
            update_images_around(current_index)

    for folder_name, images in images_by_folder.items():
        if images:
            display_images_for_folder(folder_name, images)

    # Fonction de recherche
    def filter_images():
        query = search_entry.get().lower()
        filtered_images = []
        for folder_name, images in images_by_folder.items():
            for image_path in images:
                if query in os.path.basename(image_path).lower():
                    filtered_images.append(image_path)
        return filtered_images

    # Liste des résultats
    result_listbox = tk.Listbox(frame, font=("Arial", 14), height=5)
    result_listbox.pack(pady=10, fill="x")

    def update_result_list():
        result_listbox.delete(0, tk.END)
        filtered_images = filter_images()
        for image_path in filtered_images:
            result_listbox.insert(tk.END, os.path.basename(image_path))

    search_button = tk.Button(search_frame, text="Chercher", font=("Arial", 14), command=update_result_list)
    search_button.pack(side="right", padx=5)

    def on_image_select(event):
        selected_index = result_listbox.curselection()
        if selected_index:
            selected_image = result_listbox.get(selected_index)
            for folder_name, images in images_by_folder.items():
                for image_path in images:
                    if os.path.basename(image_path) == selected_image:
                        # Mettre à jour la heatmap pour afficher l'image sélectionnée
                        # (Cette partie dépend de la structure de votre code existant)
                        # Par exemple, vous pouvez appeler une fonction pour afficher l'image dans la heatmap
                        # display_selected_image(image_path)
                        break

    result_listbox.bind('<<ListboxSelect>>', on_image_select)
