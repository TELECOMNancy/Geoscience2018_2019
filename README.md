# Geoscience2018_2019
Projet de mom geoscience à Telecom Nancy pour l'année 2018 2019

# Lancer les scripts réseaux
	lancer le script dans projet/spark_algorithms/run_clustering.sh

# Format des données (réseau)

## Input
	i, p0, p1, p2, t, norm0, norm1, norm2, e

## Output
	i0, p0, p1, p2, e

# Format des données:

## Output
	x,y,z(barycentre),temps(temps le plus petit),magnitude(somme des energie puis formule)

# Configuration pyspark

Téléchargement de pyspark

```
pip install pyspark
```

Mettre les variables d'environnement à jour
```
SPARK_HOME=<>/lib/site-packages/pyspark
HADOOP_HOME=<>/lib/site-packages/pyspark
PATH=$PATH:<>/lib/site-packages/pyspark/bin
PYSPARK_PYTHON=<>/python
PYSPARK_DRIVER_PYTHON=<>/python
```

