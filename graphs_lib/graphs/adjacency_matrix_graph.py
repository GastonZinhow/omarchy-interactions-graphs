"""
Implementação de grafo usando matriz de adjacência.
"""

import csv
from collections import deque
from typing import List, Optional, Dict
import logging

from .abstract_graph import AbstractGraph
from .exceptions import EdgeNotFoundError

logger = logging.getLogger(__name__)

class AdjacencyMatrixGraph(AbstractGraph):
    """
    Grafo direcionado ponderado implementado com matriz de adjacência.

    A matriz M é tal que M[i][j] representa o peso da aresta i -> j.
    Usamos float('inf') para indicar ausência de aresta.

    Complexidade de espaço: O(V²)
    Complexidade de verificação de aresta: O(1)

    Attributes:
        mtx (List[List[float]]): Matriz de adjacência.
    """

    def __init__(self, vertices_num: int):
        """
        Inicializa o grafo com matriz de adjacência.
        
        Args:
            vertices_num: Número de vértices no grafo.
        """
        super().__init__(vertices_num)
        self.mtx: List[List[float]] = [
            [float("inf") for _ in range(vertices_num)] 
            for _ in range(vertices_num)
        ]
        logger.info(f"AdjacencyMatrixGraph criado com {vertices_num} vértices")

    def print_graph(self) -> None:
        """Imprime a matriz de adjacência de forma legível."""
        mtx_len = len(self.vertices)
        
        print("\n" + "="*60)
        print("MATRIZ DE ADJACÊNCIA")
        print("="*60)
        print("\n     ", end="")
        
        # Cabeçalho
        for i in range(mtx_len):
            print(f"{self.vertices[i].index:5d}", end="")
        print()
        
        # Linhas da matriz
        for i in range(mtx_len):
            print(f"{self.vertices[i].index:3d}  ", end="")
            for j in range(mtx_len):
                v = self.mtx[i][j]
                if v == float("inf"):
                    print("    .", end="")
                else:
                    print(f"{v:5.1f}", end="")
            print()
        
        print("="*60 + "\n")

    def get_edge_count(self) -> int:
        """
        Conta o número total de arestas no grafo.
        
        Returns:
            Número de arestas.
        
        Complexidade: O(V²)
        """
        edge_count = 0
        
        for i in range(self.vertices_num):
            for j in range(self.vertices_num):
                if i != j and self.mtx[i][j] != float("inf"):
                    edge_count += 1
        
        return edge_count

    def has_edge(self, u: int, v: int) -> bool:
        """
        Verifica se existe aresta de u para v.
        
        Complexidade: O(1)
        """
        self.check_vertices(u, v)
        return self.mtx[u - 1][v - 1] != float("inf")

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Adiciona aresta de u para v (idempotente).
        
        Se a aresta já existe, não faz nada.
        
        Complexidade: O(1)
        """
        self.check_vertices(u, v)
        
        if self.mtx[u - 1][v - 1] == float("inf"):
            self.mtx[u - 1][v - 1] = weight
            logger.debug(f"Aresta adicionada: ({u}, {v}) com peso {weight}")
        else:
            logger.warning(f"Aresta ({u}, {v}) já existe, operação ignorada")

    def remove_edge(self, u: int, v: int) -> None:
        """
        Remove aresta de u para v (idempotente).
        
        Complexidade: O(1)
        """
        self.check_vertices(u, v)
        
        if self.mtx[u - 1][v - 1] != float("inf"):
            self.mtx[u - 1][v - 1] = float("inf")
            logger.debug(f"Aresta removida: ({u}, {v})")
        else:
            logger.warning(f"Aresta ({u}, {v}) não existe, nada a remover")

    def is_successor(self, u: int, v: int) -> bool:
        """Verifica se u é sucessor de v."""
        return self.has_edge(v, u)

    def is_predecessor(self, u: int, v: int) -> bool:
        """Verifica se u é predecessor de v."""
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Verifica se duas arestas são divergentes."""
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        
        return (
            self.has_edge(u1, v1) and 
            self.has_edge(u2, v2) and 
            u1 == u2 and 
            v1 != v2
        )

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Verifica se duas arestas são convergentes."""
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        
        return (
            self.has_edge(u1, v1) and 
            self.has_edge(u2, v2) and 
            v1 == v2 and 
            u1 != u2
        )

    def get_vertex_in_degree(self, u: int) -> int:
        """
        Calcula grau de entrada do vértice u.
        
        Complexidade: O(V)
        """
        self.check_vertex(u)
        
        in_degree = 0
        for i in range(self.vertices_num):
            if i != (u - 1) and self.mtx[i][u - 1] != float("inf"):
                in_degree += 1
        
        return in_degree

    def get_vertex_out_degree(self, u: int) -> int:
        """
        Calcula grau de saída do vértice u.
        
        Complexidade: O(V)
        """
        self.check_vertex(u)
        
        out_degree = 0
        for j in range(self.vertices_num):
            if j != (u - 1) and self.mtx[u - 1][j] != float("inf"):
                out_degree += 1
        
        return out_degree

    def set_edge_weight(self, u: int, v: int, weight: float) -> None:
        """
        Define o peso de uma aresta existente.
        
        Raises:
            EdgeNotFoundError: Se a aresta não existir.
        """
        self.check_vertices(u, v)
        
        if self.mtx[u - 1][v - 1] == float("inf"):
            raise EdgeNotFoundError(u, v)
        
        self.mtx[u - 1][v - 1] = weight
        logger.debug(f"Peso da aresta ({u}, {v}) atualizado para {weight}")

    def get_edge_weight(self, u: int, v: int) -> float:
        """
        Retorna o peso de uma aresta.
        
        Raises:
            EdgeNotFoundError: Se a aresta não existir.
        """
        self.check_vertices(u, v)
        
        if self.mtx[u - 1][v - 1] == float("inf"):
            raise EdgeNotFoundError(u, v)
        
        return self.mtx[u - 1][v - 1]

    def is_incident(self, u: int, v: int, x: int) -> bool:
        """Verifica se o vértice x é incidente à aresta (u, v)."""
        self.check_vertices(u, v)
        self.check_vertex(x)
        
        return self.has_edge(u, v) and (x == u or x == v)

    def is_empty_graph(self) -> bool:
        """Verifica se o grafo não tem arestas."""
        return self.get_edge_count() == 0

    def is_complete_graph(self) -> bool:
        """Verifica se o grafo é completo."""
        for i in range(self.vertices_num):
            for j in range(self.vertices_num):
                if i != j and self.mtx[i][j] == float("inf"):
                    return False
        return True

    def is_connected(self) -> bool:
        """
        Verifica se o grafo é fracamente conexo usando BFS.
        
        Complexidade: O(V²)
        """
        if self.vertices_num == 0:
            return True
        
        visited = [False] * self.vertices_num
        queue = deque([0])
        visited[0] = True
        
        while queue:
            u = queue.popleft()
            
            for v in range(self.vertices_num):
                if u != v and not visited[v]:
                    # Verifica aresta em qualquer direção (grafo não-direcionado implícito)
                    if (self.mtx[u][v] != float("inf") or 
                        self.mtx[v][u] != float("inf")):
                        visited[v] = True
                        queue.append(v)
        
        return all(visited)

    def export_to_gephi(
        self, 
        path: str, 
        label_map: Optional[Dict[int, str]] = None
    ) -> None:
        """Exporta grafo para CSV compatível com Gephi."""
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target", "peso"])
                
                for i in range(self.vertices_num):
                    for j in range(self.vertices_num):
                        if i != j and self.mtx[i][j] != float("inf"):
                            src = label_map.get(i + 1, i + 1) if label_map else i + 1
                            tgt = label_map.get(j + 1, j + 1) if label_map else j + 1
                            writer.writerow([src, tgt, self.mtx[i][j]])
            
            logger.info(f"Grafo exportado para Gephi: {path}")
            print(f"✓ Exportado para: {path}")
        
        except IOError as e:
            logger.error(f"Erro ao exportar para Gephi: {e}")
            raise