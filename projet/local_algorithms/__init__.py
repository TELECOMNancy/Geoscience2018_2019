# Put the files that are working in local there
from .clustering import cluster
from .magnitude import analyse_magnitude
from .dimension_kdtree import analyse_dimension
from .Analyse_temporelle import analyse_temporelle

__all__ = ['cluster', 'analyse_magnitude', 'analyse_dimension', 'analyse_temporelle']
