# 🤖 Analyse de trajectoire robot via OpenCV et ArUco

Ce projet de traitement d'images permet de suivre et d'analyser la trajectoire d'un robot en temps réel grâce à la détection d'un Tag ArUco via une caméra.

## 🛠️ Technologies utilisées
- **Python** & **OpenCV** (Traitement d'image)
- **NumPy** (Calculs de matrice)
- **Git & GitHub** (Hebergement et branches de projets)
- **Scrum / Trello** (Gestion de projet)

## ⚙️ Fonctionnement
1. **Capture** : Récupération du flux vidéo de la caméra.
2. **Détection** : Identification du marqueur ArUco posé sur le robot et des 4 autres marqueurs ArUco fixe.
3. **Tracking** : Enregistrement des coordonnées (X,Y) du centre du marqueur au fil du temps.
4. **Visualisation** : Tracé en temps réel du chemin parcouru.

## 🚀 Installation

```bash
pip install -r requirements.txt
python script.py