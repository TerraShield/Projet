import os
import numpy as np
from PIL import Image, ImageTk
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import tkinter as tk
from tkinter import ttk
from detail_image import show_image_detail
from sklearn.metrics.pairwise import cosine_similarity

def setup_cluster_page(frame, selected_folder):
    """Configure une page de clustering avec des options de filtre et de paramètres."""
    image_folder = selected_folder

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

    # Variables to store the last entered values for DBSCAN parameters
    last_eps = tk.StringVar(value="0.5")
    last_min_samples = tk.StringVar(value="5")
    last_metric = tk.StringVar(value="euclidean")
    last_algorithm = tk.StringVar(value="auto")
    last_leaf_size = tk.StringVar(value="30")
    last_p = tk.StringVar(value="None")
    last_n_jobs = tk.StringVar(value="None")

    # Fonction pour appliquer le clustering et afficher les résultats
    def apply_clustering(algorithm, clusters_count, eps=0.5, min_samples=5, metric='euclidean', algorithm_dbscan='auto', leaf_size=30, p=None, n_jobs=None):
        for widget in frame.winfo_children():
            if widget not in [filter_frame, reload_button, eps_label, eps_entry, min_samples_label, min_samples_entry, metric_label, metric_entry, algorithm_label, algorithm_entry, leaf_size_label, leaf_size_entry, p_label, p_entry, n_jobs_label, n_jobs_entry]:
                widget.destroy()

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
            tk.Label(frame, text="Algorithme non supporté.", font=("Arial", 16), fg="red").pack(pady=10)
            return

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
            tk.Label(frame, text=f"Erreur lors du clustering : {e}", font=("Arial", 16), fg="red").pack(pady=10)
            return

        # Afficher les clusters
        notebook = ttk.Notebook(frame)
        notebook.pack(fill="both", expand=True)

        unique_clusters = np.unique(all_clusters)
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

            cluster_images = [image_paths[i] for i in range(len(image_paths)) if all_clusters[i] == cluster_id]
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
    param_entry = tk.Entry(filter_frame, font=("Arial", 14))

    eps_label = tk.Label(filter_frame, text="Eps (DBSCAN) :", font=("Arial", 14))
    eps_entry = tk.Entry(filter_frame, font=("Arial", 14))

    min_samples_label = tk.Label(filter_frame, text="Min Samples (DBSCAN) :", font=("Arial", 14))
    min_samples_entry = tk.Entry(filter_frame, font=("Arial", 14))

    metric_label = tk.Label(filter_frame, text="Metric (DBSCAN) :", font=("Arial", 14))
    metric_entry = tk.Entry(filter_frame, font=("Arial", 14))

    algorithm_label = tk.Label(filter_frame, text="Algorithm (DBSCAN) :", font=("Arial", 14))
    algorithm_entry = tk.Entry(filter_frame, font=("Arial", 14))

    leaf_size_label = tk.Label(filter_frame, text="Leaf Size (DBSCAN) :", font=("Arial", 14))
    leaf_size_entry = tk.Entry(filter_frame, font=("Arial", 14))

    p_label = tk.Label(filter_frame, text="P (DBSCAN) :", font=("Arial", 14))
    p_entry = tk.Entry(filter_frame, font=("Arial", 14))

    n_jobs_label = tk.Label(filter_frame, text="N Jobs (DBSCAN) :", font=("Arial", 14))
    n_jobs_entry = tk.Entry(filter_frame, font=("Arial", 14))

    def open_dbscan_config():
        """Ouvre une fenêtre pour configurer les paramètres de DBSCAN."""
        config_window = tk.Toplevel(frame)
        config_window.title("Configuration DBSCAN")
        config_window.geometry("400x600")

        tk.Label(config_window, text="Eps :", font=("Arial", 14)).pack(pady=5)
        eps_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_eps)
        eps_entry.pack(pady=5)

        tk.Label(config_window, text="Min Samples :", font=("Arial", 14)).pack(pady=5)
        min_samples_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_min_samples)
        min_samples_entry.pack(pady=5)

        tk.Label(config_window, text="Metric :", font=("Arial", 14)).pack(pady=5)
        metric_menu = ttk.Combobox(config_window, textvariable=last_metric, values=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'precomputed'], font=("Arial", 14))
        metric_menu.pack(pady=5)

        tk.Label(config_window, text="Algorithm :", font=("Arial", 14)).pack(pady=5)
        algorithm_menu = ttk.Combobox(config_window, textvariable=last_algorithm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], font=("Arial", 14))
        algorithm_menu.pack(pady=5)

        tk.Label(config_window, text="Leaf Size :", font=("Arial", 14)).pack(pady=5)
        leaf_size_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_leaf_size)
        leaf_size_entry.pack(pady=5)

        tk.Label(config_window, text="P :", font=("Arial", 14)).pack(pady=5)
        p_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_p)
        p_entry.pack(pady=5)

        tk.Label(config_window, text="N Jobs :", font=("Arial", 14)).pack(pady=5)
        n_jobs_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_n_jobs)
        n_jobs_entry.pack(pady=5)

        def apply_dbscan_config():
            last_eps.set(eps_entry.get() or "0.5")
            last_min_samples.set(min_samples_entry.get() or "5")
            last_metric.set(metric_menu.get() or "euclidean")
            last_algorithm.set(algorithm_menu.get() or "auto")
            last_leaf_size.set(leaf_size_entry.get() or "30")
            last_p.set(p_entry.get() or "None")
            last_n_jobs.set(n_jobs_entry.get() or "None")

            eps = float(last_eps.get())
            min_samples = int(last_min_samples.get())
            metric = last_metric.get()
            algorithm = last_algorithm.get()
            leaf_size = int(last_leaf_size.get())
            p = float(last_p.get()) if last_p.get().lower() != 'none' and last_p.get() else None
            n_jobs = int(last_n_jobs.get()) if last_n_jobs.get().lower() != 'none' and last_n_jobs.get() else None

            apply_clustering('DBSCAN', 0, eps, min_samples, metric, algorithm, leaf_size, p, n_jobs)
            config_window.destroy()

        tk.Button(config_window, text="Appliquer", font=("Arial", 14), command=apply_dbscan_config).pack(pady=20)

    def update_dbscan_params(*args):
        if algo_var.get() == 'DBSCAN':
            param_label.pack_forget()
            param_entry.pack_forget()
            tk.Button(filter_frame, text="Configurer DBSCAN", font=("Arial", 14), command=open_dbscan_config).pack(side="left", padx=5, pady=2)
        else:
            param_label.pack(side="left", padx=5, pady=2)
            param_entry.pack(side="left", padx=5, pady=2)
            for widget in filter_frame.winfo_children():
                if widget.cget("text") == "Configurer DBSCAN":
                    widget.pack_forget()

    algo_var.trace('w', update_dbscan_params)

    param_label.pack(side="left", padx=5, pady=2)
    param_entry.pack(side="left", padx=5, pady=2)

    reload_button = tk.Button(filter_frame, text="Appliquer", font=("Arial", 14),
                               command=lambda: apply_clustering(
                                   algo_var.get(), int(param_entry.get() or 5), float(eps_entry.get() or 0.5), int(min_samples_entry.get() or 5),
                                   metric_entry.get() or 'euclidean', algorithm_entry.get() or 'auto', int(leaf_size_entry.get() or 30),
                                   float(p_entry.get()) if p_entry.get().lower() != 'none' and p_entry.get() else None, 
                                   int(n_jobs_entry.get()) if n_jobs_entry.get().lower() != 'none' and n_jobs_entry.get() else None
                               ))
    reload_button.pack(side="left", padx=5, pady=2)



