"""
Pacote de implementações de grafos.
"""

from .abstract_graph import AbstractGraph, Vertex
from .adjacency_list_graph import AdjacencyListGraph
from .adjacency_matrix_graph import AdjacencyMatrixGraph
from .exceptions import (
GraphException,
InvalidVertexError,
SelfLoopError,
EdgeNotFoundError,
DuplicateEdgeError,
NegativeWeightError
)

__all__ = [
'abstract_graph',
'Vertex',
'adjacency_list_graph',
'adjacency_matrix_graph',
'GraphException',
'InvalidVertexError',
'SelfLoopError',
'EdgeNotFoundError',
'DuplicateEdgeError',
'NegativeWeightError'
]

__version__ = '1.0.0'