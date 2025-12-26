# Face Blur Project

## Objectif

- Comprendre comment une image est manipulée comme une matrice de pixels  
- Utiliser OpenCV pour détecter des visages  
- Utiliser PIL pour appliquer un flou uniquement sur les zones détectées  
- Réinjecter les zones floutées dans l’image originale  

---

## 🧠 Principe

1. L’image est chargée sous forme de matrice NumPy  
2. Elle est convertie en niveaux de gris  
3. Un modèle Haar Cascade détecte les visages  
4. Chaque visage est découpé via ses coordonnées `(x, y, w, h)`  
5. La zone du visage est floutée  
6. La zone floutée est replacée dans l’image  
7. L’image finale est affichée  

---

## 🧰 Technologies

- Python  
- OpenCV  
- PIL (Pillow)  
- NumPy  
- Matplotlib  

---

