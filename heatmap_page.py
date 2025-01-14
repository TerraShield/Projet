import os
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from heatmap_algo import load_and_preprocess_images, extract_sift_features, create_bow, get_bow_histograms, dbscan_clustering, visualize_results

def setup_heatmap_page(frame, selected_folder):
    """Configure un onglet pour afficher les heatmaps des images."""
    if not os.path.exists(selected_folder):
        error_label = tk.Label(frame, text=f"Le dossier '{selected_folder}' est introuvable.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

    # Récupérer toutes les images du dossier
    image_files = [os.path.join(selected_folder, f) for f in os.listdir(selected_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_files:
        no_images_label = tk.Label(frame, text="Aucune image disponible dans le dossier sélectionné.", font=("Arial", 16), fg="red")
        no_images_label.pack(pady=20)
        return

    # Fonction pour afficher le heatmap d'une image
    def show_heatmap(original_image_path):
        images, labels, image_paths = load_and_preprocess_images(image_files)

        # Extraire les descripteurs SIFT
        keypoints_list, descriptors_list = extract_sift_features(images)

        # Créer le modèle Bag of Words
        n_words = min(50, len(descriptors_list))  # Le nombre de "mots visuels" que nous souhaitons
        bow_model = create_bow(descriptors_list, n_words=n_words)

        # Calcul des histogrammes BoW pour chaque image
        histograms = get_bow_histograms(descriptors_list, bow_model)

        # Normalisation des histogrammes
        scaler = StandardScaler()
        histograms_scaled = scaler.fit_transform(histograms)

        # Réduire la dimensionnalité pour la visualisation
        n_components = min(2, histograms_scaled.shape[0], histograms_scaled.shape[1])
        pca = PCA(n_components=n_components)
        histograms_pca = pca.fit_transform(histograms_scaled)

        # Clustering avec DBSCAN sur les histogrammes BoW
        clusters = dbscan_clustering(histograms_scaled)

        # Trouver l'indice de l'image originale
        original_index = image_paths.index(original_image_path)

        # Visualisation des résultats
        if histograms_pca.shape[1] == 1:
            histograms_pca = np.hstack((histograms_pca, np.zeros((histograms_pca.shape[0], 1))))
        visualize_results(histograms_pca, clusters, image_paths, original_index)

    # Créer un canvas avec scroll
    canvas = tk.Canvas(frame)
    scrollbar = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = tk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>",
        lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    # Afficher les boutons et les images pour chaque fichier image
    for image_file in image_files:
        try:
            img = Image.open(image_file).resize((100, 100), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            frame_row = tk.Frame(scrollable_frame)
            frame_row.pack(pady=5, fill="x")

            img_label = tk.Label(frame_row, image=photo)
            img_label.image = photo  # Préserver une référence pour éviter le garbage collector
            img_label.pack(side="left", padx=5)

            button = tk.Button(frame_row, text=os.path.basename(image_file), command=lambda path=image_file: show_heatmap(path), font=("Arial", 12))
            button.pack(side="left", padx=5)

        except Exception as e:
            print(f"Erreur lors du chargement de l'image {image_file}: {e}")
