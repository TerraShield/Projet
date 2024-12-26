from menu import create_app_window  # type: ignore

def main():
    # Créer la fenêtre de l'application en utilisant la fonction importée
    root = create_app_window()

    # Lancer la boucle principale de l'application
    root.mainloop()

# Vérifier si le script est exécuté directement
if __name__ == "__main__":
    main()
