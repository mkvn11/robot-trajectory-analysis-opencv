# PROJET D'ÉTUDE : TRACKING d'un robot avec ARUCO

import cv2
import numpy as np
import sys
import csv
import time
import os

# --- CONFIGURATION DU DOSSIER DE SAUVEGARDE ---
# Le 'r' avant les guillemets est obligatoire pour que Windows lise bien les '\'
dossier_sauvegarde = r"C:\Users\keflo\OneDrive\Bureau\ISIB\MA1\Q1 + Q2\Bureau projet d'étude"
nom_fichier = "trajectoire_robot.csv"

# os.path.join va coller le dossier et le fichier ensemble avec le bon slash
chemin_complet = os.path.join(dossier_sauvegarde, nom_fichier)

# On prépare le fichier dans ton dossier spécifique
with open(chemin_complet, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "X_mm", "Y_mm"])

# --- CONFIGURATION ---
ANCHOR_TAGS = {
    20: (600, 1400),
    21: (2400, 1400),
    23: (2400, 600),
    22: (600, 600)
}
ANCHOR_IDS = set(ANCHOR_TAGS.keys())
FIXED_POINT = (1500, 800)

def get_center(corners):
    c = corners[0]                  # Donne les 4 coins du point central
    cx = int(np.mean(c[:, 0]))      # Calcul du centre horizontal (mean = moyenne; c[:, 0] = X des 4 coins)
    cy = int(np.mean(c[:, 1]))      # Calcul du centre vertical (c[:, 1] = Y des 4 coins)
    return (cx, cy)

def draw_visuals(img, matrix_inv):
    # Dessin des Axes (0,0) qui suivent le mouvement de la table
    axis_real = np.float32([[0,0], [500,0], [0,500]]).reshape(-1, 1, 2)
    axis_pixel = cv2.perspectiveTransform(axis_real, matrix_inv)
    
    origin = tuple(axis_pixel[0][0].astype(int))
    pt_x = tuple(axis_pixel[1][0].astype(int))
    pt_y = tuple(axis_pixel[2][0].astype(int))

    cv2.arrowedLine(img, origin, pt_x, (0, 0, 255), 3)
    cv2.arrowedLine(img, origin, pt_y, (0, 255, 0), 3)
    cv2.putText(img, "Origine", origin, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Dessin du point de vérification (1500, 800)
    fixed_px = cv2.perspectiveTransform(np.array([[[FIXED_POINT[0], FIXED_POINT[1]]]], dtype="float32"), matrix_inv)
    fx, fy = int(fixed_px[0][0][0]), int(fixed_px[0][0][1])
    cv2.circle(img, (fx, fy), 10, (255, 0, 0), 2)
    cv2.putText(img, "FIXE", (fx+15, fy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

# --- INITIALISATION ---
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Remets 2 si c'est ta cam USB
if not cap.isOpened(): sys.exit("Erreur caméra")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), 
                                   cv2.aruco.DetectorParameters())

# La "Mémoire" (Dernière bonne matrice connue)
last_homography = None
last_inverse = None

# --- MÉMOIRE DE LA TRAJECTOIRE ---
historique_pixels = []  # Pour dessiner sur l'écran
historique_mm = []      # Pour l'extraction de données (si le prof le demande)
LONGUEUR_TRAIL = 100    # Nombre de points mémorisés (la longueur de la "queue" du robot)

print("MODE CONTINU : Bougez la caméra, les axes resteront collés à la table !")

while True:
    ret, frame = cap.read()
    if not ret: break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Amélioration du contraste (Filtre CLAHE)
    # Ça fait ressortir les carrés noirs même dans la pénombre
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Filtre de netteté (Rend les bords des tags ultra-tranchants)
    noyau_nettete = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    gray = cv2.filter2D(gray, -1, noyau_nettete)
    
    corners, ids, rejected = detector.detectMarkers(gray)

    if ids is not None:
        ids = ids.flatten()
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        
        # --- ETAPE 1 : RECHERCHE PERMANENTE DES ANCRES ---
        detected_anchors = {}
        robot_indices = []

        for i, tag_id in enumerate(ids):
            if tag_id in ANCHOR_IDS:
                detected_anchors[tag_id] = get_center(corners[i])
            elif tag_id == 2 or 7:
                robot_indices.append(i)

        # Si on voit les 4 ancres MAINTENANT -> On met à jour la matrice (Recalibration instantanée)
        if len(detected_anchors) == 4:
            pts_src = []
            pts_dst = []
            for tag_id in ANCHOR_TAGS:
                pts_src.append(detected_anchors[tag_id])
                pts_dst.append(ANCHOR_TAGS[tag_id])
            
            # Mise à jour des matrices
            last_homography = cv2.getPerspectiveTransform(np.array(pts_src, dtype="float32"), np.array(pts_dst, dtype="float32"))
            last_inverse = np.linalg.inv(last_homography)
            
            # Petit indicateur visuel (Carré vert en haut à gauche = Calibré en direct)
            cv2.rectangle(frame, (10, 10), (30, 30), (0, 255, 0), -1)
        else:
            # Indicateur Rouge = On utilise la mémoire (un tag est caché)
            cv2.rectangle(frame, (10, 10), (30, 30), (0, 0, 255), -1)

        # --- ETAPE 2 : TRACKING (Si on a une matrice, même ancienne) ---
        if last_homography is not None:
            # 1. On dessine les axes qui "collent" à la table
            draw_visuals(frame, last_inverse)

            # 2. On calcule la position du robot
            for i in robot_indices:
                c = get_center(corners[i])                  # 'c' contient le point en pixels trouvé par la caméra (ex: 840, 512)
                
                # ÉTAPE 1 (L'emballage) ET ÉTAPE 2 (La transformation magique) FUSIONNÉES :
                pt_mm = cv2.perspectiveTransform(np.array([[[c[0], c[1]]]], dtype="float32"), last_homography)
                # ÉTAPE 3 (Le déballage et la récupération en millimètres) :
                x, y = int(pt_mm[0][0][0]), int(pt_mm[0][0][1])

                # --- ENREGISTREMENT ET TRAJECTOIRE ---
                # On ajoute le point en pixels pour le dessin, et en mm pour les données
                historique_pixels.append(c)
                historique_mm.append((x, y))

                # On ouvre le fichier en mode 'a' (Append = Ajouter à la fin)
                with open(chemin_complet, mode='a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([time.time(), x, y])

                # On coupe la liste si elle est trop longue (efface les plus vieux points)
                if len(historique_pixels) > LONGUEUR_TRAIL:
                    historique_pixels.pop(0)
                    historique_mm.pop(0)

                # Dessin de la ligne de trajectoire (en rose par exemple)
                if len(historique_pixels) > 1:
                    cv2.polylines(frame, [np.array(historique_pixels)], isClosed=False, color=(200, 0, 200), thickness=3)
                # -------------------------------------------------

                cv2.circle(frame, c, 5, (0, 255, 255), -1)
                cv2.putText(frame, f"X:{x} Y:{y}", (c[0], c[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                print(f"Robot: {x}, {y}")

    cv2.imshow("Tracking Continu", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()