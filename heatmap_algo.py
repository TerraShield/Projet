try:
    from skimage.io import imread
    from skimage.transform import resize
except ImportError:
    raise ImportError("Le module 'skimage' n'est pas installé. Veuillez l'installer en utilisant 'pip install scikit-image'.")

try:
    import cv2  # OpenCV pour SIFT
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
from sklearn.cluster import KMeans  # Pour le Bag of Words
import matplotlib.image as mpimg
from sklearn.metrics.pairwise import cosine_similarity
import os

# 1. Chargement et prétraitement des images
def load_and_preprocess_images(image_paths, target_size=(64, 64)):
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

# 2. Extraction des descripteurs SIFT
def extract_sift_features(images):
    sift = cv2.SIFT_create()  # Crée un détecteur SIFT
    all_keypoints = []
    all_descriptors = []
    
    for img in images:
        # Convertit l'image en format approprié pour OpenCV (BGR)
        img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        
        # Détecte les keypoints et calcule les descripteurs SIFT
        keypoints, descriptors = sift.detectAndCompute(img_bgr, None)
        
        if descriptors is not None:
            all_keypoints.append(keypoints)
            all_descriptors.append(descriptors)
        else:
            # Si aucun descripteur n'est trouvé, on met un tableau vide
            all_keypoints.append([])
            all_descriptors.append(np.array([]))
    
    return all_keypoints, all_descriptors

# 3. Création du modèle Bag of Words (BoW)
def create_bow(descriptors_list, n_words=50):
    # Fusionner tous les descripteurs de toutes les images
    all_descriptors = np.vstack(descriptors_list)
    
    # Utiliser KMeans pour créer les "mots visuels"
    kmeans = KMeans(n_clusters=n_words, random_state=42)
    kmeans.fit(all_descriptors)
    
    return kmeans

# 4. Représenter chaque image par un histogramme de mots visuels
def get_bow_histograms(descriptors_list, bow_model):
    histograms = []
    
    for descriptors in descriptors_list:
        # Attribuer chaque descripteur à un mot visuel
        words = bow_model.predict(descriptors)
        
        # Créer un histogramme de fréquence des mots visuels
        histogram = np.zeros(len(bow_model.cluster_centers_))
        for word in words:
            histogram[word] += 1
        
        histograms.append(histogram)
    
    return np.array(histograms)

# 5. Clustering avec DBSCAN sur les histogrammes BoW
def dbscan_clustering(histograms_scaled):
    dbscan = DBSCAN(eps=0.3, min_samples=2, metric='euclidean')  # Paramètres à ajuster selon les données
    clusters = dbscan.fit_predict(histograms_scaled)
    return clusters

# 6. Visualisation des résultats
def visualize_results(histograms_pca, clusters, image_paths, original_index):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # a) Scatter plot du clustering
    unique_clusters = set(clusters)
    colors = plt.cm.get_cmap('viridis', len(unique_clusters))
    for cluster in unique_clusters:
        cluster_points = histograms_pca[clusters == cluster]
        if cluster == -1:
            color = 'red'  # Noise points
        else:
            color = colors(cluster)
        axes[0].scatter(cluster_points[:, 0], cluster_points[:, 1], color=color, label=f'Cluster {cluster}')
    
    # Highlight the original image
    axes[0].scatter(histograms_pca[original_index, 0], histograms_pca[original_index, 1], color='blue', edgecolor='black', s=100, label='Original')

    axes[0].set_title("Clustering des lettrines avec DBSCAN")
    axes[0].set_xlabel("Feature 1")
    axes[0].set_ylabel("Feature 2")
    axes[0].legend()
    cbar = fig.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=axes[0])
    cbar.set_label("Cluster")

    # b) Matrice de similarité entre les histogrammes BoW
    similarity_matrix = cosine_similarity(histograms_pca)
    im = axes[1].imshow(similarity_matrix, cmap='plasma', aspect='auto')
    axes[1].set_title("Matrice de similarité")
    axes[1].set_xlabel("Index des lettrines")
    axes[1].set_ylabel("Index des lettrines")
    cbar2 = fig.colorbar(im, ax=axes[1])
    cbar2.set_label("Similarité")

    # Fonction de clic pour afficher l'image
    click_threshold = 1.2  # Ajuste ce seuil selon tes besoins

    def on_click(event):
        if event.inaxes == axes[0]:  # Vérifie si le clic est dans l'axe du graphique PCA
            x_click, y_click = event.xdata, event.ydata
            
            # Calcul des distances entre le clic et tous les points
            distances = np.sqrt((histograms_pca[:, 0] - x_click)**2 + (histograms_pca[:, 1] - y_click)**2)
            
            # Trouve l'indice du point le plus proche
            nearest_point_idx = np.argmin(distances)
            
            # Vérifie si la distance est inférieure au seuil (clic "sur" le point)
            if distances[nearest_point_idx] < click_threshold:
                # Affiche l'image associée à ce point
                img = mpimg.imread(image_paths[nearest_point_idx])  # Charge l'image
                plt.figure(figsize=(6, 6))
                plt.imshow(img, cmap='gray')
                plt.title(f"Image : {image_paths[nearest_point_idx]}")
                plt.axis('off')  # Désactive les axes pour l'affichage de l'image
                plt.show()

    # Connecte la fonction de clic à l'événement
    fig.canvas.mpl_connect('button_press_event', on_click)

    # Ajout d'annotations interactives avec mplcursors
    cursor = mplcursors.cursor(im, hover=True)

    @cursor.connect("add")
    def on_add(sel):
        # Récupération des coordonnées (lignes et colonnes) de la cellule survolée
        i, j = int(sel.target[0]), int(sel.target[1])
        # Ajout des informations sur les lettrines comparées et la similarité
        sel.annotation.set_text(f"Similarité: {similarity_matrix[i, j]:.2f}")
        sel.annotation.set_backgroundcolor('white')  # Optionnel : rendre le fond blanc pour plus de lisibilité

    # Fonction pour activer le zoom avec la molette, centré sur le curseur
    def zoom(event):
        ax = event.inaxes  # Récupère les axes actuels
        if ax is None:
            return
        
        # Obtenir les coordonnées de la position de la souris en termes de données de l'axe
        x_mouse, y_mouse = event.xdata, event.ydata
        
        # Si les coordonnées sont invalides (ex. si la souris est en dehors du graphe), ne pas zoomer
        if x_mouse is None or y_mouse is None:
            return
        
        # Obtenir les limites actuelles des axes
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        
        # Calcul du facteur de zoom en fonction de la direction de la molette
        scale_factor = 2.0 # Augmenter cette valeur pour un zoom plus rapide
        if event.button == 'up':  # Zoom avant (molette vers le haut)
            factor = 1 / scale_factor
        elif event.button == 'down':  # Zoom arrière (molette vers le bas)
            factor = scale_factor
        else:
            return
        
        # Calcul des nouvelles limites centrées sur la souris
        width = xlim[1] - xlim[0]
        height = ylim[1] - ylim[0]
        
        # Nouvelles limites
        new_width = width * factor
        new_height = height * factor
        
        # Calcul des nouvelles limites x et y en centrant autour du curseur
        new_xlim = [x_mouse - new_width * (x_mouse - xlim[0]) / width, x_mouse + new_width * (xlim[1] - x_mouse) / width]
        new_ylim = [y_mouse - new_height * (y_mouse - ylim[0]) / height, y_mouse + new_height * (ylim[1] - y_mouse) / height]
        
        # Appliquer les nouvelles limites
        ax.set_xlim(new_xlim)
        ax.set_ylim(new_ylim)
        fig.canvas.draw()  # Redessine le graphique avec les nouvelles limites

    # Connecte l'événement de zoom à la fonction
    fig.canvas.mpl_connect('scroll_event', zoom)

    # Ajustement des espaces entre les graphiques
    plt.tight_layout()
    plt.show()
