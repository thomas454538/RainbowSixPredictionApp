# Prédiction des Victoires dans Rainbow Six Siege

Ce projet utilise un modèle d'ensemble d'arbres de décision pour prédire si le nombre de victoires d'un joueur est au-dessus de la moyenne. L'approche d'ensemble permet de combiner plusieurs modèles pour obtenir de meilleures performances et une plus grande robustesse.

## Modèle Utilisé : Ensemble d'Arbres de Décision

Le modèle est un **ensemble d'arbres de décision**. Voici les étapes principales de sa construction :

1. **Bootstrap** : Pour chaque arbre de décision, un sous-échantillon aléatoire (bootstrap) est tiré de l'ensemble d'entraînement, avec remise. Cela permet de diversifier les arbres et de rendre l'ensemble plus robuste.
2. **Entraînement d'un Arbre** : Chaque arbre de décision est entraîné sur un sous-échantillon avec une profondeur maximale limitée pour éviter le surapprentissage. La profondeur maximale est réglée pour maintenir un bon compromis entre précision et généralisation.
3. **Prédiction par Vote Majoritaire** : Une fois tous les arbres entraînés, chaque arbre effectue une prédiction pour chaque échantillon. Les prédictions finales sont faites en utilisant le vote majoritaire parmi tous les arbres de l'ensemble.
4. **Évaluation** : Le modèle est évalué en utilisant des métriques de précision, de F1, et une matrice de confusion pour mesurer les performances en classification.

## Données

- Source : [Kaggle](https://www.kaggle.com/datasets/fahadalqahtani/tom-clancys-rainbow-six-siege)
- Variables : Statistiques de jeu incluant les kills, deaths, xp, etc.


## Résultats

- Précision finale (Test) : 98%
- Score F1 final (Test) : 98%

Le modèle et les graphiques sont sauvegardés dans le dossier de travail.

