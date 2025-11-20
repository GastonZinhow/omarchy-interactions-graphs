"""
Exceções customizadas para operações de grafos.
"""

class GraphException(Exception):
    """Exceção base para todas as exceções relacionadas a grafos."""
pass

class InvalidVertexError(GraphException):
    """Exceção lançada quando um vértice inválido é referenciado."""

def __init__(self, vertex: int, max_vertices: int):
    self.vertex = vertex
    self.max_vertices = max_vertices
    message = (
        f"Vértice {vertex} é inválido. "
        f"Deve estar entre 1 e {max_vertices}."
    )
    super().__init__(message)

class SelfLoopError(GraphException):
    """Exceção lançada ao tentar criar um laço (aresta de um vértice para ele mesmo)."""

def __init__(self, vertex: int):
    self.vertex = vertex
    message = f"Laços não são permitidos. Tentativa de criar aresta ({vertex}, {vertex})."
    super().__init__(message)

class EdgeNotFoundError(GraphException):
    """Exceção lançada quando uma aresta não existe."""

def __init__(self, u: int, v: int):
    self.u = u
    self.v = v
    message = f"Aresta ({u}, {v}) não existe no grafo."
    super().__init__(message)

class DuplicateEdgeError(GraphException):
    """Exceção lançada ao tentar adicionar uma aresta duplicada."""

def __init__(self, u: int, v: int):
    self.u = u
    self.v = v
    message = f"Aresta ({u}, {v}) já existe no grafo."
    super().__init__(message)

class NegativeWeightError(GraphException):
    """Exceção lançada ao tentar usar peso negativo em operações que não o suportam."""

def __init__(self, weight: float):
    self.weight = weight
    message = f"Peso negativo ({weight}) não é permitido nesta operação."
    super().__init__(message)