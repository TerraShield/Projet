import tkinter as tk
from tkinter import ttk, filedialog
from page2 import create_page2  # Fonction pour gérer la page 2
from cluster_page import setup_cluster_page  # Fonction pour gérer la page Clustering
from histogramme import setup_histogram_page  # Fonction pour gérer la page Histogramme
from heatmap_page import setup_heatmap_page  # Fonction pour gérer la page Heatmap

def create_app_window():
    """Crée la fenêtre principale de l'application avec un Notebook.

    Returns:
        tk.Tk: La fenêtre principale de l'application.
    """
    root = tk.Tk()
    root.title("Application d'Images")
    root.geometry("1280x720")  # Taille de la fenêtre principale

    # Créer une variable pour stocker le dossier sélectionné
    selected_folder = tk.StringVar()

    # Fonction pour sélectionner un dossier
    def select_folder():
        """Ouvre une boîte de dialogue pour sélectionner un dossier contenant des images."""
        folder = filedialog.askdirectory(title="Sélectionnez un dossier contenant des images")
        if folder:
            selected_folder.set(folder)
            refresh_tabs()  # Rafraîchir les onglets après la sélection

    # Fonction pour rafraîchir les onglets `Liste`, `Clustering`, `Histogramme` et `Heatmap`
    def refresh_tabs():
        """Rafraîchit les onglets `Liste`, `Clustering`, `Histogramme` et `Heatmap` après la sélection d'un dossier."""
        for frame, setup_function in [(frame2, create_page2), (frame3, setup_cluster_page), (frame4, setup_histogram_page), (frame5, setup_heatmap_page)]:
            for widget in frame.winfo_children():
                widget.destroy()
            if setup_function == create_page2:
                setup_function(frame, root, selected_folder.get())
            else:
                setup_function(frame, selected_folder.get())

    # Création du Notebook pour les onglets
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Styliser les onglets
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Arial", 14, "bold"))

    # Onglet 1 : Menu principal
    frame1 = tk.Frame(notebook)
    notebook.add(frame1, text="Menu")
    tk.Label(frame1, text="Bienvenue dans le Menu Principal", font=("Arial", 24, "bold")).pack(pady=20)
    tk.Button(frame1, text="Sélectionner un dossier", command=select_folder, font=("Arial", 14)).pack(pady=10)
    tk.Label(frame1, textvariable=selected_folder, font=("Arial", 12), fg="blue").pack(pady=5)

    # Onglet 2 : Liste des images
    frame2 = tk.Frame(notebook)
    notebook.add(frame2, text="Liste")
    create_page2(frame2, root, selected_folder.get())  # Passe le dossier sélectionné

    # Onglet 3 : Clustering
    frame3 = tk.Frame(notebook)
    notebook.add(frame3, text="Clustering")
    setup_cluster_page(frame3, selected_folder.get())  # Passe le dossier sélectionné

    # Onglet 4 : Histogrammes
    frame4 = tk.Frame(notebook)
    notebook.add(frame4, text="Histogramme")
    setup_histogram_page(frame4, selected_folder.get())  # Configure la page Histogramme

    # Onglet 5 : Heatmap
    frame5 = tk.Frame(notebook)
    notebook.add(frame5, text="Heatmap")
    setup_heatmap_page(frame5, selected_folder.get())  # Configure la page Heatmap

    return root

