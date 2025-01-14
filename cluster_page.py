import os
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk
from detail_image import show_image_detail

def setup_cluster_page(frame, selected_folder):
    """Configure une page de clustering avec des options de filtre."""
    image_folder = selected_folder if selected_folder else "images"

    # Récupérer les images par sous-dossier
    image_paths = [os.path.join(root_dir, file)
                   for root_dir, _, files in os.walk(image_folder)
                   for file in files if file.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_paths:
        error_label = tk.Label(frame, text="Aucune image trouvée dans le dossier sélectionné.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    # Charger et prétraiter les images
    def load_and_preprocess_images(image_paths, target_size=(64, 64)):
        images = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
                images.append(np.array(img).flatten())
            except Exception as e:
                print(f"Erreur lors du chargement de {path}: {e}")
        return np.array(images)

    images = load_and_preprocess_images(image_paths)

    # Normaliser les images
    scaler = StandardScaler()
    images_scaled = scaler.fit_transform(images)

    # Réduire la dimensionnalité pour le clustering
    pca = PCA(n_components=min(50, images_scaled.shape[0], images_scaled.shape[1]))
    images_pca = pca.fit_transform(images_scaled)

    # Fonction pour appliquer le clustering et afficher les résultats
    def apply_clustering(n_clusters, improve=False):
        # Clear existing clusters but keep filter options and reload button
        for widget in frame.winfo_children():
            if widget not in [filter_frame, reload_button, cluster_label, cluster_entry]:
                widget.destroy()

        # Utiliser des caractéristiques basées sur les couleurs et les PCA pour le clustering
        color_features = []
        for path in image_paths:
            try:
                img = Image.open(path).convert("RGB").resize((32, 32), Image.Resampling.LANCZOS)
                color_features.append(np.array(img).flatten())
            except Exception as e:
                color_features.append(np.zeros((32, 32, 3)).flatten())
                print(f"Erreur lors de l'extraction des caractéristiques de {path}: {e}")

        color_features = np.array(color_features)
        color_features_scaled = scaler.fit_transform(color_features)

        combined_features = np.hstack((color_features_scaled, images_pca))

        # Appliquer le clustering avec scikit-learn
        clustering_model = KMeans(n_clusters=n_clusters, random_state=42, n_init=20 if improve else 10, max_iter=500 if improve else 300)
        clusters = clustering_model.fit_predict(combined_features)

        # Vérifier le nombre de clusters distincts trouvés
        unique_clusters = len(set(clusters))
        print(f"Number of distinct clusters found: {unique_clusters}")

        # Créer un Notebook pour les clusters
        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True)

        # Afficher les images par cluster
        for cluster_id in range(unique_clusters):
            cluster_frame = tk.Frame(notebook)
            notebook.add(cluster_frame, text=f"Cluster {cluster_id + 1}")

            canvas = tk.Canvas(cluster_frame)
            scrollbar = tk.Scrollbar(cluster_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = tk.Frame(canvas)

            scrollable_frame.bind(
                "<Configure>",
                lambda e, canvas=canvas: canvas.configure(scrollregion=canvas.bbox("all"))
            )

            canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            canvas.configure(yscrollcommand=scrollbar.set)

            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            # Sort images within the cluster based on similarity
            cluster_images = [image_paths[i] for i in range(len(image_paths)) if clusters[i] == cluster_id]
            cluster_features = [combined_features[i] for i in range(len(image_paths)) if clusters[i] == cluster_id]

            if cluster_features:
                similarity_matrix = cosine_similarity(cluster_features)
                sorted_indices = np.argsort(-similarity_matrix[0])  # Sort by similarity to the first image

                row, col = 0, 0
                for idx in sorted_indices:
                    image_path = cluster_images[idx]
                    try:
                        img = Image.open(image_path).resize((100, 100), Image.Resampling.LANCZOS)
                        photo = ImageTk.PhotoImage(img)

                        label = tk.Label(scrollable_frame, image=photo)
                        label.image = photo  # Préserver une référence pour éviter le garbage collector
                        label.grid(row=row, column=col, padx=10, pady=10)

                        # Associer un clic gauche pour ouvrir l'image avec le lecteur par défaut
                        label.bind("<Button-1>", lambda e, path=image_path: show_image_detail(path))

                        col += 1
                        if col > 4:  # 5 images par ligne
                            col = 0
                            row += 1

                    except Exception as e:
                        print(f"Erreur lors du chargement de l'image {image_path}: {e}")

            # Lier la molette de la souris pour le défilement
            def on_mouse_wheel(event, canvas=canvas):
                canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

            frame.bind_all("<MouseWheel>", on_mouse_wheel)

    # Ajouter des options de filtre
    filter_frame = tk.Frame(frame)
    filter_frame.pack(pady=10, fill="x")

    cluster_label = tk.Label(filter_frame, text="Nombre de clusters :", font=("Arial", 14))
    cluster_label.pack(side="left", padx=5)

    cluster_var = tk.IntVar(value=5)
    cluster_entry = tk.Entry(filter_frame, textvariable=cluster_var, font=("Arial", 14))
    cluster_entry.pack(side="left", padx=5)

    # Ajouter un bouton de rechargement
    reload_button = tk.Button(filter_frame, text="Recharger", font=("Arial", 14), command=lambda: apply_clustering(cluster_var.get(), improve=True))
    reload_button.pack(side="left", padx=5)

    # Appliquer le clustering initialement
    apply_clustering(cluster_var.get())
