"""
Implementação de grafo usando lista de adjacência.
"""

import csv
from collections import deque
from typing import Dict, List, Tuple, Optional
import logging

from .abstract_graph import AbstractGraph
from .exceptions import EdgeNotFoundError, DuplicateEdgeError

logger = logging.getLogger(__name__)

class AdjacencyListGraph(AbstractGraph):
    """
    Grafo direcionado ponderado implementado com lista de adjacência.

    A lista de adjacência armazena, para cada vértice, uma lista de tuplas
    (destino, peso) representando as arestas de saída.

    Complexidade de espaço: O(V + E)
    Complexidade de verificação de aresta: O(grau(v))

    Attributes:
        adj_list (Dict[int, List[Tuple[int, float]]]): Dicionário de listas de adjacência.
    """

    def __init__(self, vertices_num: int):
        """
        Inicializa o grafo com lista de adjacência vazia.
        
        Args:
            vertices_num: Número de vértices no grafo.
        """
        super().__init__(vertices_num)
        self.adj_list: Dict[int, List[Tuple[int, float]]] = {
            v.index: [] for v in self.vertices
        }
        logger.info(f"AdjacencyListGraph criado com {vertices_num} vértices")

    def print_graph(self) -> None:
        """Imprime a lista de adjacência de forma legível."""
        print("\n" + "="*60)
        print("LISTA DE ADJACÊNCIA")
        print("="*60)
        
        for v in self.vertices:
            edges = ", ".join(
                f"({dest}, w={w:.2f})" 
                for dest, w in self.adj_list[v.index]
            )
            print(f"Vértice {v.index:3d}: [{edges}]")
        print("="*60 + "\n")

    def get_edge_count(self) -> int:
        """
        Conta o número total de arestas no grafo.
        
        Returns:
            Número de arestas.
        
        Complexidade: O(V)
        """
        count = sum(len(neighbors) for neighbors in self.adj_list.values())
        return count

    def has_edge(self, u: int, v: int) -> bool:
        """
        Verifica se existe aresta de u para v.
        
        Args:
            u: Vértice de origem.
            v: Vértice de destino.
        
        Returns:
            True se a aresta existe.
        
        Complexidade: O(grau(u))
        """
        self.check_vertices(u, v)
        return any(dest == v for dest, _ in self.adj_list[u])

    def add_edge(self, u: int, v: int, weight: float = 1.0) -> None:
        """
        Adiciona aresta de u para v (idempotente).
        
        Se a aresta já existe, não faz nada (comportamento idempotente).
        
        Args:
            u: Vértice de origem.
            v: Vértice de destino.
            weight: Peso da aresta (padrão: 1.0).
        
        Raises:
            InvalidVertexError: Se u ou v forem inválidos.
            SelfLoopError: Se u == v.
        
        Complexidade: O(grau(u))
        """
        self.check_vertices(u, v)
        
        if not self.has_edge(u, v):
            self.adj_list[u].append((v, weight))
            logger.debug(f"Aresta adicionada: ({u}, {v}) com peso {weight}")
        else:
            logger.warning(f"Aresta ({u}, {v}) já existe, operação ignorada")

    def remove_edge(self, u: int, v: int) -> None:
        """
        Remove aresta de u para v (idempotente).
        
        Se a aresta não existe, não faz nada.
        
        Args:
            u: Vértice de origem.
            v: Vértice de destino.
        
        Complexidade: O(grau(u))
        """
        self.check_vertices(u, v)
        
        original_len = len(self.adj_list[u])
        self.adj_list[u] = [
            (dest, w) for dest, w in self.adj_list[u] if dest != v
        ]
        
        if len(self.adj_list[u]) < original_len:
            logger.debug(f"Aresta removida: ({u}, {v})")
        else:
            logger.warning(f"Aresta ({u}, {v}) não existe, nada a remover")

    def is_successor(self, u: int, v: int) -> bool:
        """Verifica se u é sucessor de v (existe v -> u)."""
        self.check_vertices(u, v)
        return self.has_edge(v, u)

    def is_predecessor(self, u: int, v: int) -> bool:
        """Verifica se u é predecessor de v (existe u -> v)."""
        self.check_vertices(u, v)
        return self.has_edge(u, v)

    def is_divergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Verifica se duas arestas são divergentes (mesma origem)."""
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        
        return (
            self.has_edge(u1, v1) and 
            self.has_edge(u2, v2) and 
            u1 == u2 and 
            v1 != v2
        )

    def is_convergent(self, u1: int, v1: int, u2: int, v2: int) -> bool:
        """Verifica se duas arestas são convergentes (mesmo destino)."""
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
        
        Complexidade: O(V + E)
        """
        self.check_vertex(u)
        
        in_degree = 0
        for src in self.adj_list:
            for dest, _ in self.adj_list[src]:
                if dest == u:
                    in_degree += 1
        
        return in_degree

    def get_vertex_out_degree(self, u: int) -> int:
        """
        Calcula grau de saída do vértice u.
        
        Complexidade: O(1)
        """
        self.check_vertex(u)
        return len(self.adj_list[u])

    def set_edge_weight(self, u: int, v: int, weight: float) -> None:
        """
        Define o peso de uma aresta existente.
        
        Raises:
            EdgeNotFoundError: Se a aresta não existir.
        """
        self.check_vertices(u, v)
        
        for i, (dest, _) in enumerate(self.adj_list[u]):
            if dest == v:
                self.adj_list[u][i] = (v, weight)
                logger.debug(f"Peso da aresta ({u}, {v}) atualizado para {weight}")
                return
        
        raise EdgeNotFoundError(u, v)

    def get_edge_weight(self, u: int, v: int) -> float:
        """
        Retorna o peso de uma aresta.
        
        Raises:
            EdgeNotFoundError: Se a aresta não existir.
        """
        self.check_vertices(u, v)
        
        for dest, weight in self.adj_list[u]:
            if dest == v:
                return weight
        
        raise EdgeNotFoundError(u, v)

    def is_incident(self, u: int, v: int, x: int) -> bool:
        """
        Verifica se o vértice x é incidente à aresta (u, v).
        
        Args:
            u, v: Aresta a verificar.
            x: Vértice para testar incidência.
        
        Returns:
            True se x == u ou x == v e a aresta existe.
        """
        self.check_vertices(u, v)
        self.check_vertex(x)
        
        return self.has_edge(u, v) and (x == u or x == v)

    def is_empty_graph(self) -> bool:
        """Verifica se o grafo não tem arestas."""
        return self.get_edge_count() == 0

    def is_complete_graph(self) -> bool:
        """
        Verifica se o grafo é completo.
        
        Um grafo direcionado é completo se existe aresta entre
        todo par de vértices distintos (em ambas as direções).
        """
        num_v = self.vertices_num
        
        for u in self.adj_list:
            neighbors = set(dest for dest, _ in self.adj_list[u])
            expected = set(v for v in range(1, num_v + 1) if v != u)
            
            if neighbors != expected:
                return False
        
        return True

    def is_connected(self) -> bool:
        """
        Verifica se o grafo é fracamente conexo usando BFS.
        
        Ignora direção das arestas para verificar conexidade.
        
        Complexidade: O(V + E)
        """
        if self.vertices_num == 0:
            return True
        
        visited = set([1])
        queue = deque([1])
        
        while queue:
            u = queue.popleft()
            
            # Vizinhos de saída
            for dest, _ in self.adj_list[u]:
                if dest not in visited:
                    visited.add(dest)
                    queue.append(dest)
            
            # Vizinhos de entrada
            for src in self.adj_list:
                if any(dest == u for dest, _ in self.adj_list[src]):
                    if src not in visited:
                        visited.add(src)
                        queue.append(src)
        
        return len(visited) == self.vertices_num

    def export_to_gephi(
        self, 
        path: str, 
        label_map: Optional[Dict[int, str]] = None
    ) -> None:
        """
        Exporta grafo para CSV compatível com Gephi.
        
        Args:
            path: Caminho do arquivo de saída.
            label_map: Mapeamento opcional de índices para rótulos.
        
        Exemplo de CSV gerado:
            source,target,peso
            1,2,3.5
            2,3,1.0
        """
        try:
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["source", "target", "peso"])
                
                for u in self.adj_list:
                    for dest, weight in self.adj_list[u]:
                        src_label = label_map.get(u, u) if label_map else u
                        dest_label = label_map.get(dest, dest) if label_map else dest
                        writer.writerow([src_label, dest_label, weight])
            
            logger.info(f"Grafo exportado para Gephi: {path}")
            print(f"✓ Exportado para: {path}")
        
        except IOError as e:
            logger.error(f"Erro ao exportar para Gephi: {e}")
            raise