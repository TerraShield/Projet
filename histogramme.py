import tkinter as tk

def setup_histogram_page(frame):
    """Configure un onglet vide pour les histogrammes."""
    label = tk.Label(frame, text="Cette page est vide pour le moment.", font=("Arial", 16), fg="gray")
    label.pack(expand=True, pady=20)
