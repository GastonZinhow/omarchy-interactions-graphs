from abc import ABC, abstractmethod
from typing import List, Optional
import logging

from .exceptions import InvalidVertexError, SelfLoopError

logger = logging.getLogger(__name__)

class Vertex:
    def __init__(self, index: int, label: str = "", weight: float = 0.0):
        if index < 1:
            raise ValueError(f"Índice do vértice deve ser >= 1, recebido: {index}")
        self.index = index
        self.label = label
        self.weight = weight

    def __repr__(self) -> str:
        return f"Vertex(index={self.index}, label='{self.label}', weight={self.weight})"

    def __eq__(self, other) -> bool:
        if not isinstance(other, Vertex):
            return False
        return self.index == other.index

    def __hash__(self) -> int:
        return hash(self.index)

class AbstractGraph(ABC):
    def __init__(self, vertices_num: int):
        if vertices_num < 0:
            raise ValueError(
                f"Número de vértices deve ser não-negativo, recebido: {vertices_num}"
            )
        self.vertices_num = vertices_num
        self.vertices = [Vertex(i) for i in range(1, vertices_num + 1)]
        logger.info(f"Grafo inicializado com {vertices_num} vértices")

    def check_vertex(self, u: int) -> bool:
        if u < 1 or u > self.vertices_num:
            raise InvalidVertexError(u, self.vertices_num)
        return True

    def check_vertices(self, u: int, v: int) -> None:
        self.check_vertex(u)
        self.check_vertex(v)
        if u == v:
            raise SelfLoopError(u)

    def get_vertex_count(self) -> int:
        return self.vertices_num

    @abstractmethod
    def get_edge_count(self) -> int:
        pass

    @abstractmethod
    def has_edge(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        pass

    @abstractmethod
    def remove_edge(self, u: int, v: int) -> None:
        pass

    @abstractmethod
    def is_successor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def is_predecessor(self, u: int, v: int) -> bool:
        pass

    @abstractmethod
    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        pass

    @abstractmethod
    def get_vertex_in_degree(self, u: int) -> int:
        pass

    @abstractmethod
    def get_vertex_out_degree(self, u: int) -> int:
        pass

    def set_vertex_weight(self, u: int, weight: float) -> None:
        self.check_vertex(u)
        self.vertices[u - 1].weight = weight
        logger.debug(f"Peso do vértice {u} definido para {weight}")

    def get_vertex_weight(self, u: int) -> float:
        self.check_vertex(u)
        return self.vertices[u - 1].weight

    @abstractmethod
    def set_edge_weight(self, u: int, v: int, weight: float) -> None:
        pass

    @abstractmethod
    def get_edge_weight(self, u: int, v: int) -> float:
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        pass

    @abstractmethod
    def is_empty_graph(self) -> bool:
        pass

    @abstractmethod
    def is_complete_graph(self) -> bool:
        pass

    @abstractmethod
    def export_to_gephi(self, path: str, label_map: Optional[dict] = None) -> None:
        pass