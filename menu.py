import tkinter as tk
from tkinter import ttk
from page2 import create_page2  # Fonction pour gérer la page 2
from heatmap_page import setup_heatmap_page  # Fonction pour gérer la page Heatmap

def create_app_window():
    """Crée la fenêtre principale de l'application avec un Notebook."""
    root = tk.Tk()
    root.title("Application d'Images")
    root.geometry("1280x720")  # Taille de la fenêtre principale

    # Création du Notebook pour les onglets
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Styliser les onglets
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Arial", 14, "bold"))

    # Onglet 1 : Menu principal
    frame1 = tk.Frame(notebook)
    notebook.add(frame1, text="Menu")
    menu_label = tk.Label(frame1, text="Bienvenue dans le Menu Principal", font=("Arial", 24, "bold"))
    menu_label.pack(pady=50)

    # Onglet 2 : Liste des images
    frame2 = tk.Frame(notebook)
    notebook.add(frame2, text="Liste")
    create_page2(frame2, root)  # Configure la page 2 pour afficher la liste des images

    # Onglet 3 : Heatmaps
    frame3 = tk.Frame(notebook)
    notebook.add(frame3, text="Heatmap")
    setup_heatmap_page(frame3, image_folder="images")  # Corrige le nom de la fonction appelée

    return root

def main():
    """Fonction principale pour lancer l'application."""
    root = create_app_window()
    root.mainloop()

if __name__ == "__main__":
    main()
