import os
from tkinter import filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk
from detail_image import show_image_detail
from cluster_algo import load_and_preprocess_images, apply_clustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def setup_cluster_page(frame, selected_folder):
    """
    Configure une page de clustering avec des options de filtre et de paramètres.

    Args:
        frame (tk.Frame): Le frame parent où l'interface sera construite
        selected_folder (str): Le chemin du dossier contenant les images à analyser

    Notes:
        - Crée une interface complète pour le clustering d'images
        - Charge et prétraite automatiquement les images du dossier sélectionné
        - Applique une réduction de dimensionnalité PCA
        - Propose des options de configuration pour KMeans et DBSCAN
    """
    image_folder = selected_folder

    # Récupérer les images par sous-dossier
    image_paths = [os.path.join(root_dir, file)
                   for root_dir, _, files in os.walk(image_folder)
                   for file in files if file.endswith(('.jpg', '.png', '.jpeg'))]

    if not image_paths:
        error_label = tk.Label(frame, text="Aucune image trouvée dans le dossier sélectionné.", font=("Arial", 16), fg="red")
        error_label.pack(pady=20)
        return

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

    def display_clusters(all_clusters):
        """
    Affiche les clusters dans une interface à onglets et ajoute un bouton de téléchargement.

    Args:
        all_clusters (numpy.array): Tableau des labels de clusters pour chaque image

    Notes:
        - Crée un notebook avec un onglet par cluster
        - Pour chaque cluster :
            * Crée un canvas scrollable
            * Affiche les miniatures des images (100x100 pixels)
            * Permet le zoom par clic sur les images
        - Ajoute un bouton de téléchargement des clusters
        
    Structure de l'interface:
        - Un onglet par cluster identifié
        - Grille de 5 images par ligne
        - Scrollbar vertical pour naviguer dans les images
        - Bouton de téléchargement en bas
        
    Interactions:
        - Clic sur image : ouvre la vue détaillée
        - Clic sur bouton télécharger : lance le téléchargement des clusters
        
    Gestion des erreurs:
        - Capture les erreurs de chargement d'images individuelles
        - Continue l'affichage même si certaines images échouent
    """
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

    def apply_clustering_wrapper(algorithm, clusters_count, eps=0.5, min_samples=5, metric='euclidean', algorithm_dbscan='auto', leaf_size=30, p=None, n_jobs=None):
        """
        Wrapper pour appliquer l'algorithme de clustering sélectionné.

        Args:
            algorithm (str): L'algorithme à utiliser ('KMeans' ou 'DBSCAN')
            clusters_count (int): Nombre de clusters pour KMeans
            eps (float): Paramètre eps pour DBSCAN (default: 0.5)
            min_samples (int): Nombre minimum d'échantillons pour DBSCAN (default: 5)
            metric (str): Métrique de distance pour DBSCAN (default: 'euclidean')
            algorithm_dbscan (str): Algorithme de calcul des voisins (default: 'auto')
            leaf_size (int): Taille des feuilles pour les arbres (default: 30)
            p (float): Paramètre de puissance pour la métrique Minkowski
            n_jobs (int): Nombre de jobs parallèles
        """
        for widget in frame.winfo_children():
            if widget not in [filter_frame, reload_button, eps_label, eps_entry, min_samples_label, min_samples_entry, metric_label, metric_entry, algorithm_label, algorithm_entry, leaf_size_label, leaf_size_entry, p_label, p_entry, n_jobs_label, n_jobs_entry]:
                widget.destroy()

        try:
            all_clusters = apply_clustering(images_pca, image_paths, algorithm, clusters_count, eps, min_samples, metric, algorithm_dbscan, leaf_size, p, n_jobs)
            display_clusters(all_clusters)
        except RuntimeError as e:
            tk.Label(frame, text=str(e), font=("Arial", 16), fg="red").pack(pady=10)
    
        # Ajoutez cette fonction dans setup_cluster_page
    def download_clusters(all_clusters):
        """
    Télécharge les clusters générés dans un dossier sélectionné par l'utilisateur.

    Args:
        all_clusters (numpy.array): Tableau des labels de clusters pour chaque image

    Notes:
        - Crée un sous-dossier pour chaque cluster
        - Copie les images dans leurs dossiers de cluster respectifs
        - Affiche une confirmation ou une erreur via messagebox
    """
        # Ouvre une boîte de dialogue pour sélectionner le dossier de destination
        destination_folder = filedialog.askdirectory(title="Sélectionner un dossier pour enregistrer les clusters")
        if not destination_folder:
            return  # Si aucun dossier n'est sélectionné, on annule l'opération

        try:
            # Crée des sous-dossiers pour chaque cluster
            unique_clusters = np.unique(all_clusters)
            for cluster_id in unique_clusters:
                cluster_name = f"Cluster_{cluster_id}" if cluster_id != -1 else "Unclustered"
                cluster_folder = os.path.join(destination_folder, cluster_name)
                os.makedirs(cluster_folder, exist_ok=True)

                cluster_images = [image_paths[i] for i in range(len(image_paths)) if all_clusters[i] == cluster_id]
                for image_path in cluster_images:
                    image_name = os.path.basename(image_path)
                    destination_path = os.path.join(cluster_folder, image_name)
                    try:
                        Image.open(image_path).save(destination_path)  # Copie l'image dans le sous-dossier
                    except Exception as e:
                        print(f"Erreur lors de l'enregistrement de l'image {image_path}: {e}")

            messagebox.showinfo("Succès", f"Les clusters ont été enregistrés dans le dossier : {destination_folder}")
        except Exception as e:
            messagebox.showerror("Erreur", f"Une erreur s'est produite lors de l'enregistrement des clusters : {e}")

    # Ajoutez un bouton pour télécharger les clusters après l'affichage des résultats
    def display_clusters(all_clusters):
        """
    Affiche les clusters dans une interface à onglets.

    Args:
        all_clusters (numpy.array): Tableau des labels de clusters pour chaque image

    Notes:
        - Crée un onglet pour chaque cluster
        - Affiche les miniatures des images dans une grille scrollable
        - Permet de cliquer sur les images pour les voir en détail
        - Inclut un bouton pour télécharger les clusters
    """
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

        # Ajoutez un bouton pour télécharger les clusters
        download_button = tk.Button(frame, text="Télécharger les clusters", font=("Arial", 14),
                                     command=lambda: download_clusters(all_clusters))
        download_button.pack(pady=10)

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

    def create_tooltip(widget, text):
        """
    Crée une infobulle pour un widget.

    Args:
        widget (tk.Widget): Le widget qui recevra l'infobulle
        text (str): Le texte à afficher dans l'infobulle
    """
        tooltip = tk.Toplevel(widget)
        tooltip.wm_overrideredirect(True)
        tooltip.wm_geometry("+0+0")
        label = tk.Label(tooltip, text=text, background="yellow", relief="solid", borderwidth=1, font=("Arial", 10))
        label.pack()
        tooltip.withdraw()

        def enter(event):
            x, y, _, _ = widget.bbox("insert")
            x += widget.winfo_rootx() + 25
            y += widget.winfo_rooty() + 25
            tooltip.wm_geometry(f"+{x}+{y}")
            tooltip.deiconify()

        def leave(event):
            tooltip.withdraw()

        widget.bind("<Enter>", enter)
        widget.bind("<Leave>", leave)

    def open_dbscan_config():
        """
    Ouvre une fenêtre de configuration pour les paramètres DBSCAN.

    Notes:
        - Permet de configurer tous les paramètres de DBSCAN
        - Inclut des infobulles explicatives pour chaque paramètre
        - Conserve les dernières valeurs entrées
    """
        config_window = tk.Toplevel(frame)
        config_window.title("Configuration DBSCAN")
        config_window.geometry("400x600")

        tk.Label(config_window, text="Eps :", font=("Arial", 14)).pack(pady=5)
        eps_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_eps)
        eps_entry.pack(pady=5)
        create_tooltip(eps_entry, "Distance maximale entre deux échantillons pour qu'ils soient considérés comme voisins.")

        tk.Label(config_window, text="Min Samples :", font=("Arial", 14)).pack(pady=5)
        min_samples_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_min_samples)
        min_samples_entry.pack(pady=5)
        create_tooltip(min_samples_entry, "Nombre minimum d'échantillons dans un voisinage pour qu'un point soit considéré comme un noyau.")

        tk.Label(config_window, text="Metric :", font=("Arial", 14)).pack(pady=5)
        metric_menu = ttk.Combobox(config_window, textvariable=last_metric, values=['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'precomputed'], font=("Arial", 14))
        metric_menu.pack(pady=5)
        create_tooltip(metric_menu, "La métrique utilisée pour calculer la distance entre les points.")

        tk.Label(config_window, text="Algorithm :", font=("Arial", 14)).pack(pady=5)
        algorithm_menu = ttk.Combobox(config_window, textvariable=last_algorithm, values=['auto', 'ball_tree', 'kd_tree', 'brute'], font=("Arial", 14))
        algorithm_menu.pack(pady=5)
        create_tooltip(algorithm_menu, "L'algorithme utilisé pour calculer les plus proches voisins.")

        tk.Label(config_window, text="Leaf Size :", font=("Arial", 14)).pack(pady=5)
        leaf_size_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_leaf_size)
        leaf_size_entry.pack(pady=5)
        create_tooltip(leaf_size_entry, "Taille des feuilles passées à BallTree ou KDTree.")

        tk.Label(config_window, text="P :", font=("Arial", 14)).pack(pady=5)
        p_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_p)
        p_entry.pack(pady=5)
        create_tooltip(p_entry, "Le paramètre de puissance pour la métrique Minkowski.")

        tk.Label(config_window, text="N Jobs :", font=("Arial", 14)).pack(pady=5)
        n_jobs_entry = tk.Entry(config_window, font=("Arial", 14), textvariable=last_n_jobs)
        n_jobs_entry.pack(pady=5)
        create_tooltip(n_jobs_entry, "Le nombre de tâches à utiliser pour le calcul.")

        def apply_dbscan_config():
            """
    Applique la configuration DBSCAN et lance le clustering.
    
    Notes:
        - Récupère toutes les valeurs des champs de configuration
        - Convertit les valeurs en types appropriés (float, int)
        - Gère les valeurs 'None' pour les paramètres optionnels
        - Lance le clustering avec les paramètres configurés
        - Ferme la fenêtre de configuration après application
    
    Effets de bord:
        - Met à jour les variables last_* avec les nouvelles valeurs
        - Déclenche l'exécution du clustering
        - Ferme la fenêtre de configuration
    """
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

            apply_clustering_wrapper('DBSCAN', 0, eps, min_samples, metric, algorithm, leaf_size, p, n_jobs)
            config_window.destroy()

        tk.Button(config_window, text="Appliquer", font=("Arial", 14), command=apply_dbscan_config).pack(pady=20)

    def update_dbscan_params(*args):
        """
    Met à jour l'interface en fonction de l'algorithme sélectionné.
    
    Args:
        *args: Arguments variables (non utilisés mais nécessaires pour le callback de trace)
    
    Notes:
        - Appelé automatiquement lorsque l'algorithme change
        - Si DBSCAN est sélectionné :
            * Cache les paramètres de KMeans
            * Affiche le bouton de configuration DBSCAN
        - Si KMeans est sélectionné :
            * Affiche le paramètre de nombre de clusters
            * Cache les éléments de configuration DBSCAN
    
    Effets de bord:
        - Modifie la visibilité des widgets dans l'interface
    """
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
                               command=lambda: apply_clustering_wrapper(
                                   algo_var.get(), int(param_entry.get() or 5), float(eps_entry.get() or 0.5), int(min_samples_entry.get() or 5),
                                   metric_entry.get() or 'euclidean', algorithm_entry.get() or 'auto', int(leaf_size_entry.get() or 30),
                                   float(p_entry.get()) if p_entry.get().lower() != 'none' and p_entry.get() else None, 
                                   int(n_jobs_entry.get()) if n_jobs_entry.get().lower() != 'none' and n_jobs_entry.get() else None
                               ))
    reload_button.pack(side="left", padx=5, pady=2)



