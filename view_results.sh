#!/bin/bash
# Script pour ouvrir la visionneuse de résultats HTML

# Vérifier si le dossier results existe
if [ ! -d "results" ]; then
    echo "Erreur: Le dossier 'results' n'existe pas!"
    echo "Assurez-vous d'avoir entraîné au moins un modèle avant d'utiliser ce script."
    exit 1
fi

# Ouvrir le fichier HTML dans le navigateur par défaut
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    open view_results.html
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    xdg-open view_results.html
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    # Windows
    start view_results.html
else
    echo "Plateforme non reconnue. Veuillez ouvrir manuellement le fichier 'view_results.html'."
fi

echo "La visionneuse de résultats a été ouverte dans votre navigateur."
echo "Note: Si les images ne s'affichent pas, vérifiez que les chemins dans view_results.html correspondent à votre structure de dossiers." 