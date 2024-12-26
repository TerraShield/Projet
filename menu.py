import tkinter as tk
from tkinter import ttk
from page2 import create_page2  # On importe la fonction create_page2 du fichier page2.py
from heatmap_page import HeatmapPage  # On importe la classe HeatmapPage du fichier heatmap_page.py

def create_app_window():
    """Crée la fenêtre principale de l'application avec un Notebook."""
    root = tk.Tk()
    root.title("Application d'Images")

    # Redimensionner la fenêtre principale en 1280x720
    root.geometry("1280x720")

    # Créer le notebook (onglet)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Styliser le notebook
    style = ttk.Style()
    style.configure("TNotebook.Tab", font=("Arial", 14, "bold"))

    # Créer le premier onglet (menu principal)
    frame1 = tk.Frame(notebook)
    notebook.add(frame1, text="Menu")

    # Ajouter du contenu au menu principal
    menu_label = tk.Label(frame1, text="Bienvenue dans le Menu Principal", font=("Arial", 24, "bold"))
    menu_label.pack(pady=50)

    # Créer le deuxième onglet (page des images)
    frame2 = tk.Frame(notebook)
    notebook.add(frame2, text="Liste")

    # Créer la page 2 avec les images et la galerie
    create_page2(frame2, root)

    # Créer le troisième onglet (page des heatmaps)
    frame3 = tk.Frame(notebook)
    notebook.add(frame3, text="Heatmaps")

    # Créer la page des heatmaps
    HeatmapPage(frame3)

    return root

def main():
    """Fonction principale qui lance l'application."""
    root = create_app_window()
    root.mainloop()

if __name__ == "__main__":
    main()
