import tkinter as tk
from tkinter import ttk
from page2 import create_page2  # On importe la fonction create_page2 du fichier page2.py

def create_app_window():
    """Crée la fenêtre principale de l'application avec un Notebook."""
    root = tk.Tk()
    root.title("Application d'Images")

    # Redimensionner la fenêtre principale en 1980x720
    root.geometry("1280x720")

    # Créer le notebook (onglet)
    notebook = ttk.Notebook(root)
    notebook.pack(fill="both", expand=True)

    # Créer le premier onglet (menu principal)
    frame1 = tk.Frame(notebook)
    notebook.add(frame1, text="Menu")

    # Créer le deuxième onglet (page des images)
    frame2 = tk.Frame(notebook)
    notebook.add(frame2, text="Liste")
    
    # Créer la page 2 avec les images et la galerie
    create_page2(frame2)

    return root

def main():
    """Fonction principale qui lance l'application."""
    root = create_app_window()
    root.mainloop()

if __name__ == "__main__":
    main()
