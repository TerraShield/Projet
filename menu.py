import tkinter as tk
from tkinter import ttk
from page2 import create_page2  # On importe la fonction pour afficher la deuxième page

def create_app_window():
    """Crée la fenêtre principale de l'application avec un onglet."""
    
    # Créer la fenêtre principale
    root = tk.Tk()

    # Définir le titre de la fenêtre
    root.title("Application avec Onglets")
    root.geometry("600x400")

    # Créer un Notebook (pour les onglets)
    notebook = ttk.Notebook(root)

    # Créer les pages des onglets
    frame1 = tk.Frame(notebook)
    frame2 = tk.Frame(notebook)

    # Ajouter les pages au notebook
    notebook.add(frame1, text="Page d'Accueil")
    notebook.add(frame2, text="Page 2")

    # Placer le notebook dans la fenêtre
    notebook.pack(fill="both", expand=True)

    # Remplir la première page (Page d'Accueil)
    label1 = tk.Label(frame1, text="Bienvenue sur la Page d'Accueil", font=("Arial", 16))
    label1.pack(pady=20)

    # Remplir la deuxième page en appelant la fonction `create_page2`
    create_page2(frame2)  # Affiche les images dans le deuxième onglet

    return root
