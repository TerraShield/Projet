import os

# Set the number of cores to use
os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with the number of cores you want to use

from menu import create_app_window  # type: ignore

def main():
    """
    Fonction principale pour lancer l'application.

    Cette fonction crée la fenêtre principale de l'application et lance la boucle principale de l'interface utilisateur.
    """
    root = create_app_window()
    root.mainloop()

if __name__ == "__main__":
    main()


