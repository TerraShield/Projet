import os
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from detail_image import show_image_detail

def setup_cluster_page(frame, selected_folder):
    """Configure une page de clustering avec des options de filtre et de paramètres."""
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
    def apply_clustering(algorithm, clusters_count):
        for widget in frame.winfo_children():
            if widget not in [filter_frame, reload_button]:
                widget.destroy()

        # Appliquer le clustering avec l'algorithme choisi
        if algorithm == 'KMeans':
            clustering_model = KMeans(
                n_clusters=clusters_count,  # Utilise le nombre de clusters fourni
                random_state=42
            )
        elif algorithm == 'DBSCAN':
            clustering_model = DBSCAN(
                eps=0.5,  # Paramètre par défaut
                min_samples=clusters_count  # Utilise clusters_count comme min_samples
            )
        else:
            tk.Label(frame, text="Algorithme non supporté.", font=("Arial", 16), fg="red").pack(pady=10)
            return

        # Calculer les clusters
        try:
            clusters = clustering_model.fit_predict(images_pca)
        except Exception as e:
            tk.Label(frame, text=f"Erreur lors du clustering : {e}", font=("Arial", 16), fg="red").pack(pady=10)
            return

        # Afficher les clusters
        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True)

        unique_clusters = np.unique(clusters)
        for cluster_id in unique_clusters:
            cluster_frame = tk.Frame(notebook)
            notebook.add(cluster_frame, text=f"Cluster {cluster_id}" if cluster_id != -1 else "List")

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

            cluster_images = [image_paths[i] for i in range(len(image_paths)) if clusters[i] == cluster_id]
            for idx, image_path in enumerate(cluster_images):
                try:
                    img = Image.open(image_path).resize((100, 100), Image.Resampling.LANCZOS)
                    photo = ImageTk.PhotoImage(img)

                    label = tk.Label(scrollable_frame, image=photo)
                    label.image = photo
                    label.grid(row=idx // 5, column=idx % 5, padx=10, pady=10)

                    label.bind("<Button-1>", lambda e, path=image_path: show_image_detail(path))
                except Exception as e:
                    print(f"Erreur lors du chargement de l'image {image_path}: {e}")

    # Ajouter des options de paramétrage
    filter_frame = tk.Frame(frame)
    filter_frame.pack(pady=10, fill="x")

    algo_label = tk.Label(filter_frame, text="Algorithme :", font=("Arial", 14))
    algo_label.pack(side="left", padx=5)

    algo_var = tk.StringVar(value='KMeans')
    algo_menu = ttk.Combobox(filter_frame, textvariable=algo_var, values=['KMeans', 'DBSCAN'], font=("Arial", 14))
    algo_menu.pack(side="left", padx=5)

    param_label = tk.Label(filter_frame, text="Nombre de clusters :", font=("Arial", 14))
    param_label.pack(side="left", padx=5)

    param_entry = tk.Entry(filter_frame, font=("Arial", 14))
    param_entry.pack(side="left", padx=5)

    reload_button = tk.Button(filter_frame, text="Appliquer", font=("Arial", 14),
                               command=lambda: apply_clustering(
                                   algo_var.get(), int(param_entry.get())
                               ))
    reload_button.pack(side="left", padx=5)

    # Appliquer le clustering initialement
    apply_clustering('KMeans', 5)
