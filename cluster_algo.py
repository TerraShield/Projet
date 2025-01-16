import os
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

def load_and_preprocess_images(image_paths, target_size=(64, 64)):
    images = []
    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB").resize(target_size, Image.Resampling.LANCZOS)
            images.append(np.array(img).flatten())
        except Exception as e:
            print(f"Erreur lors du chargement de {path}: {e}")
    return np.array(images)

def apply_clustering(images_pca, image_paths, algorithm, clusters_count, eps=0.5, min_samples=5, metric='euclidean', algorithm_dbscan='auto', leaf_size=30, p=None, n_jobs=None):
    # Appliquer le clustering avec l'algorithme choisi
    if algorithm == 'KMeans':
        clustering_model = KMeans(
            n_clusters=clusters_count,  # Utilise le nombre de clusters fourni
            random_state=42
        )
    elif algorithm == 'DBSCAN':
        clustering_model = DBSCAN(
            eps=eps,  # Utilise le paramètre eps fourni
            min_samples=min_samples,  # Utilise le paramètre min_samples fourni
            metric=metric,
            algorithm=algorithm_dbscan,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs
        )
    else:
        raise ValueError("Algorithme non supporté.")

    # Diviser les images en morceaux gérables
    chunk_size = 1000
    num_chunks = (len(images_pca) + chunk_size - 1) // chunk_size
    all_clusters = np.zeros(len(images_pca), dtype=int)

    try:
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(images_pca))
            chunk_data = images_pca[start_idx:end_idx]

            chunk_clusters = clustering_model.fit_predict(chunk_data)
            all_clusters[start_idx:end_idx] = chunk_clusters

            if algorithm == 'DBSCAN' and -1 in chunk_clusters:
                valid_clusters = chunk_clusters != -1
                if np.any(valid_clusters):
                    distance_matrix = 1 - cosine_similarity(chunk_data)
                    for i, cluster in enumerate(chunk_clusters):
                        if cluster == -1:
                            nearest_cluster = np.argmin(distance_matrix[i][valid_clusters])
                            chunk_clusters[i] = chunk_clusters[valid_clusters][nearest_cluster]

            all_clusters[start_idx:end_idx] = chunk_clusters

    except Exception as e:
        raise RuntimeError(f"Erreur lors du clustering : {e}")

    return all_clusters
