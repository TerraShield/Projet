try:
    from skimage.io import imread
    from skimage.transform import resize
except ImportError:
    raise ImportError("Le module 'skimage' n'est pas installé. Veuillez l'installer en utilisant 'pip install scikit-image'.")

try:
    import cv2  
except ImportError:
    raise ImportError("Le module 'cv2' n'est pas installé. Veuillez l'installer en utilisant 'pip install opencv-python-headless'.")

try:
    import mplcursors
except ImportError:
    raise ImportError("Le module 'mplcursors' n'est pas installé. Veuillez l'installer en utilisant 'pip install mplcursors'.")

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
import os
from sklearn.decomposition import PCA

def load_and_preprocess_images(image_paths, target_size=(64, 64)):
    """ Chargement et prétraitement des images

    Args:
        image_paths (list): Liste de chemins d'accès aux images
        target_size (tuple, optional): Taille cible pour redimensionner les images (width, height). Defaults to (64, 64).

    Returns:
        tuple: (np.array(images), labels, image_paths)
            np.array(images) : Un tableau contenant les images prétraitées
            labels : Une liste de nom de fichiers des images
            image_paths : La liste originale des chemins d'accès
    """
    images = []
    labels = []
    for path in image_paths:
        try:
            img = imread(path, as_gray=True)  # Charge en niveaux de gris
            img_resized = resize(img, target_size)  # Redimensionne
            images.append(img_resized)
            labels.append(os.path.basename(path))  # Nom de fichier comme label
        except Exception as e:
            print(f"Erreur lors du chargement de {path}: {e}")

    return np.array(images), labels, image_paths

def extract_sift_features(images):
    """Extraction des descripteurs SIFT

    Args:
        images (np.array): Tableau contenant les images prétraitées

    Returns:
        tuple: (all_keypoints, all_descriptors)
            all_keypoints : Une liste de keypoints détectés pour chaque image
            all_descriptors : Une liste de descripteurs SIFT pour chaque image
    """
    sift = cv2.SIFT_create()  # Crée un détecteur SIFT
    all_keypoints = []
    all_descriptors = []
    
    for img in images:
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)  # Convertit l'image en format approprié pour OpenCV (BGR)
        keypoints, descriptors = sift.detectAndCompute(img_bgr, None)  # Détecte les keypoints et calcule les descripteurs SIFT
        
        all_keypoints.append(keypoints)
        all_descriptors.append(descriptors if descriptors is not None else np.array([]))
    
    return all_keypoints, all_descriptors

def create_bow(descriptors_list, n_words=50):
    """Création du modèle Bag of Words (BoW)

    Args:
        descriptors_list (list): Liste de descripteurs SIFT pour chaque image
        n_words (int, optional): Nombre de clusters (mots visuels) à créer. Defaults to 50.

    Returns:
        KMeans: Modèle KMeans entraîné sur les descripteurs
    """
    all_descriptors = np.vstack(descriptors_list)  # Fusionner tous les descripteurs de toutes les images
    kmeans = KMeans(n_clusters=n_words, random_state=42)  # Utiliser KMeans pour créer les "mots visuels"
    kmeans.fit(all_descriptors)
    return kmeans

def get_bow_histograms(descriptors_list, bow_model):
    """Représenter chaque image par un histogramme de mots visuels

    Args:
        descriptors_list (list): Liste de descripteurs SIFT pour chaque image
        bow_model (KMeans): Modèle KMeans entraîné sur les descripteurs

    Returns:
        np.array: Tableau contenant les histogrammes de mots visuels pour chaque image
    """
    histograms = []
    for descriptors in descriptors_list:
        words = bow_model.predict(descriptors)  # Attribuer chaque descripteur à un mot visuel
        histogram = np.zeros(len(bow_model.cluster_centers_))
        for word in words:
            histogram[word] += 1
        histograms.append(histogram)
    return np.array(histograms)

def dbscan_clustering(histograms_scaled, eps=0.3, min_samples=2):
    """Clustering avec DBSCAN sur les histogrammes BoW

    Args:
        histograms_scaled (np.array): Tableau contenant les histogrammes de mots visuels normalisés
        eps (float, optional): Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins. Defaults to 0.3.
        min_samples (int, optional): Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau. Defaults to 2.

    Returns:
        np.array: Tableau contenant les labels de clusters pour chaque image
    """
    similarity_matrix = cosine_similarity(histograms_scaled)  # Calculer la matrice de similarité
    distance_matrix = 1 - similarity_matrix  # Convertir la matrice de similarité en une matrice de distance
    distance_matrix[distance_matrix < 0] = 0  # S'assurer qu'il n'y a pas de valeurs négatives dans la matrice de distance
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')  # Appliquer DBSCAN en utilisant la matrice de distance
    clusters = dbscan.fit_predict(distance_matrix)
    
    # Réassigner le cluster -1 au cluster le plus proche
    if -1 in clusters:
        valid_clusters = clusters != -1
        if np.any(valid_clusters):
            for i, cluster in enumerate(clusters):
                if cluster == -1:
                    nearest_cluster = np.argmin(distance_matrix[i][valid_clusters])
                    clusters[i] = clusters[valid_clusters][nearest_cluster]
    
    return clusters

def visualize_results(histograms_pca, clusters, image_paths, original_index):
    """Visualisation des résultats

    Args:
        histograms_pca (np.array): Tableau contenant les histogrammes de mots visuels réduits en 2D
        clusters (np.array): Tableau contenant les labels de clusters pour chaque image
        image_paths (list): Liste de chemins d'accès aux images
        original_index (int): Indice de l'image originale dans la liste des chemins d'accès
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Scatter plot du clustering
    unique_clusters = set(clusters)
    colors = plt.cm.get_cmap('viridis', len(unique_clusters))
    for cluster in unique_clusters:
        cluster_points = histograms_pca[clusters == cluster]
        color = 'red' if cluster == -1 else colors(cluster)
        axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster}')
    
    axes[0].scatter(histograms_pca[original_index, 0], histograms_pca[original_index, 1], color='blue', edgecolor='black', s=100, label='Original')
    axes[0].set_title("Clustering des lettrines avec DBSCAN")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].legend()
    fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[0]).set_label("Cluster")

    # Matrice de similarité entre les histogrammes BoW
    similarity_matrix = cosine_similarity(histograms_pca)
    im = axes[1].imshow(similarity_matrix, cmap='plasma', aspect='auto')
    axes[1].set_title("Matrice de similarité")
    axes[1].set_xlabel("Index des lettrines")
    axes[1].set_ylabel("Index des lettrines")
    fig.colorbar(im, ax=axes[1]).set_label("Similarité")

    # Permet de montrer les images pointé par le clique de la souris sur les graphes
    # Sur le graphe PCA, montre l'image qui est relié au point le plus proche de la souris lors du clique
    # Sur la matrice, montre quelles images sont comparées entre elles
    def on_click(event):
        if event.inaxes == axes[0]:  # Vérifie si le clic est dans l'axe du graphique PCA
            x_click, y_click = event.xdata, event.ydata
            distances = np.sqrt((histograms_pca[:, 0] - x_click)**2 + (histograms_pca[:, 1] - y_click)**2)
            nearest_point_idx = np.argmin(distances)
            if distances[nearest_point_idx] < 1.2:  # Ajuste ce seuil selon tes besoins
                img = mpimg.imread(image_paths[nearest_point_idx])  # Charge l'image
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='gray')
                plt.title(f"Image : {image_paths[nearest_point_idx]}")
                plt.axis('off')  # Désactive les axes pour l'affichage de l'image
                plt.show()
        elif event.inaxes == axes[1]:  # Vérifie si le clic est dans l'axe de la matrice de similarité
            x_click, y_click = int(event.xdata), int(event.ydata)
            if 0 <= x_click < len(image_paths) and 0 <= y_click < len(image_paths):
                similarity = similarity_matrix[y_click, x_click]
                img1 = mpimg.imread(image_paths[y_click])
                img2 = mpimg.imread(image_paths[x_click])
                fig, ax = plt.subplots(1, 2, figsize=(12, 6))
                ax[0].imshow(img1, cmap='gray')
                ax[0].axis('off')
                ax[1].imshow(img2, cmap='gray')
                ax[1].axis('off')
                plt.suptitle(f"Similarité: {similarity:.2f}")
                plt.show()

    fig.canvas.mpl_connect('button_press_event', on_click)

    # Sur la matrice de similarité, permet de montrer le taux de différence
    # 1 | jaune = image identique
    # Plus les deux images comparés sont différentes, plus la couleur tirera sur le violet, et plus le chiffre sera bas
    cursor = mplcursors.cursor(im, hover=True)
    @cursor.connect("add")
    def on_add(sel):
        i, j = int(sel.target[0]), int(sel.target[1])
        sel.annotation.set_text(f"Similarité: {similarity_matrix[i, j]:.2f}")
        sel.annotation.set_backgroundcolor('white')

    # Permet de gérer le zoom sur les graphes
    def zoom(event):
        ax = event.inaxes
        if ax is None:
            return
        x_mouse, y_mouse = event.xdata, event.ydata
        if x_mouse is None or y_mouse is None:
            return
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        scale_factor = 2.0
        factor = 1 / scale_factor if event.button == 'up' else scale_factor
        width, height = xlim[1] - xlim[0], ylim[1] - ylim[0]
        new_width, new_height = width * factor, height * factor
        new_xlim = [x_mouse - new_width * (x_mouse - xlim[0]) / width, x_mouse + new_width * (xlim[1] - x_mouse) / width]
        new_ylim = [y_mouse - new_height * (y_mouse - ylim[0]) / height, y_mouse + new_height * (ylim[1] - y_mouse) / height]
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw()

    fig.canvas.mpl_connect('scroll_event', zoom)
    plt.tight_layout()
    plt.show()

def show_heatmap(image_path, images_by_folder):
    """Afficher une heatmap des images similaires

    Args:
        image_path (str): Chemin d'accès de l'image originale
        images_by_folder (dict): Dictionnaire contenant les chemins d'accès des images par dossier
    """
    images, _, image_paths = load_and_preprocess_images([img for imgs in images_by_folder.values() for img in imgs])

    # Extraire les descripteurs SIFT
    keypoints_list, descriptors_list = extract_sift_features(images)

    # Créer le modèle Bag of Words
    n_words = min(50, len(descriptors_list))
    bow_model = create_bow(descriptors_list, n_words=n_words)

    # Calcul des histogrammes BoW pour chaque image
    histograms = get_bow_histograms(descriptors_list, bow_model)

    # Normalisation des histogrammes
    scaler = StandardScaler()
    histograms_scaled = scaler.fit_transform(histograms)

    # Réduire la dimensionnalité pour la visualisation
    pca = PCA(n_components=2)
    histograms_pca = pca.fit_transform(histograms_scaled)

    # Clustering avec DBSCAN sur les histogrammes BoW
    clusters = dbscan_clustering(histograms_scaled)

    # Trouver l'indice de l'image originale
    original_index = image_paths.index(image_path)

    # Visualisation des résultats
    visualize_results(histograms_pca, clusters, image_paths, original_index)
