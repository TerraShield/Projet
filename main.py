from menu import create_app_window  # type: ignore

def main():
    """Fonction principale pour lancer l'application."""
    root = create_app_window()
    root.mainloop()

if __name__ == "__main__":
    main()