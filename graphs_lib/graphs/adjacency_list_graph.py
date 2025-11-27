"""
Implementação de grafo usando lista de adjacência.
"""

import csv
from collections import deque
from typing import Dict, List, Tuple, Optional, Set
import logging
import math

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
      
      Se a aresta já existe, SOMA o peso (para múltiplas interações).
      
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
      
      # Busca se já existe aresta
      for i, (dest, w) in enumerate(self.adj_list[u]):
          if dest == v:
              # Aresta já existe: soma o peso
              self.adj_list[u][i] = (v, w + weight)
              logger.debug(f"Peso da aresta ({u}, {v}) atualizado: {w} + {weight} = {w + weight}")
              return
      
      # Aresta não existe: adiciona nova
      self.adj_list[u].append((v, weight))
      logger.debug(f"Aresta adicionada: ({u}, {v}) com peso {weight}")

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

  def get_all_neighbors(self, u: int, ignore_direction: bool = False) -> Set[int]:
      """
      Retorna conjunto de vizinhos de u.
      
      Args:
          u: Vértice.
          ignore_direction: Se True, considera grafo não-direcionado.
      
      Returns:
          Conjunto de vizinhos.
      """
      self.check_vertex(u)
      neighbors = set()
      
      # Vizinhos de saída (sucessores)
      for dest, _ in self.adj_list[u]:
          neighbors.add(dest)
      
      # Se ignorar direção, adiciona vizinhos de entrada (predecessores)
      if ignore_direction:
          for src in self.adj_list:
              for dest, _ in self.adj_list[src]:
                  if dest == u:
                      neighbors.add(src)
      
      return neighbors

  def calculate_density(self) -> float:
      """
      Calcula densidade do grafo direcionado.
      
      Densidade = E / (V * (V - 1))
      
      Returns:
          Densidade entre 0 e 1.
      """
      n = self.vertices_num
      if n <= 1:
          return 0.0
      
      m = self.get_edge_count()
      return m / (n * (n - 1))

  def calculate_clustering_coefficient(self, u: int) -> float:
      """
      Calcula coeficiente de clustering LOCAL de um vértice.
      
      Para grafos direcionados: considera vizinhos (sucessores e predecessores)
      e conta triângulos fechados.
      
      Args:
          u: Vértice.
      
      Returns:
          Coeficiente entre 0 e 1.
      """
      self.check_vertex(u)
      
      # Pega vizinhos (ignorando direção)
      neighbors = self.get_all_neighbors(u, ignore_direction=True)
      k = len(neighbors)
      
      if k < 2:
          return 0.0
      
      # Conta conexões entre vizinhos
      connections = 0
      neighbors_list = list(neighbors)
      
      for i in range(len(neighbors_list)):
          for j in range(i + 1, len(neighbors_list)):
              v1 = neighbors_list[i]
              v2 = neighbors_list[j]
              
              # Verifica se existe conexão em qualquer direção
              if self.has_edge(v1, v2) or self.has_edge(v2, v1):
                  connections += 1
      
      # Coeficiente = conexões / máximo possível
      max_connections = (k * (k - 1)) / 2
      return connections / max_connections if max_connections > 0 else 0.0

  def calculate_average_clustering_coefficient(self) -> float:
      """
      Calcula coeficiente de clustering MÉDIO do grafo.
      
      Returns:
          Média dos coeficientes locais.
      """
      if self.vertices_num == 0:
          return 0.0
      
      total = sum(
          self.calculate_clustering_coefficient(v.index) 
          for v in self.vertices
      )
      
      return total / self.vertices_num

  def calculate_degree_assortativity(self) -> float:
      """
      Calcula assortatividade de grau (correlação de Pearson entre graus das arestas).
      
      Mede se nós de alto grau tendem a se conectar com outros de alto grau.
      
      Returns:
          Valor entre -1 e 1:
          - Positivo: rede assortativa (similares se conectam)
          - Negativo: rede dissassortativa (diferentes se conectam)
          - Zero: sem correlação
      """
      if self.get_edge_count() == 0:
          return 0.0
      
      # Coletar graus de origem e destino de cada aresta
      degrees_src = []
      degrees_dst = []
      
      for u in self.adj_list:
          degree_u = self.get_vertex_out_degree(u) + self.get_vertex_in_degree(u)
          
          for dest, _ in self.adj_list[u]:
              degree_v = self.get_vertex_out_degree(dest) + self.get_vertex_in_degree(dest)
              
              degrees_src.append(degree_u)
              degrees_dst.append(degree_v)
      
      n = len(degrees_src)
      if n == 0:
          return 0.0
      
      # Calcular correlação de Pearson
      mean_src = sum(degrees_src) / n
      mean_dst = sum(degrees_dst) / n
      
      numerator = sum(
          (degrees_src[i] - mean_src) * (degrees_dst[i] - mean_dst)
          for i in range(n)
      )
      
      std_src = math.sqrt(sum((x - mean_src) ** 2 for x in degrees_src) / n)
      std_dst = math.sqrt(sum((x - mean_dst) ** 2 for x in degrees_dst) / n)
      
      if std_src == 0 or std_dst == 0:
          return 0.0
      
      return numerator / (n * std_src * std_dst)

  def detect_communities_label_propagation(self, max_iter: int = 100) -> Dict[int, int]:
      """
      Detecta comunidades usando Label Propagation (simples e eficiente).
      
      Algoritmo:
      1. Cada nó começa com seu próprio label
      2. Iterativamente, cada nó adota o label mais comum entre seus vizinhos
      3. Converge quando labels param de mudar
      
      Args:
          max_iter: Número máximo de iterações.
      
      Returns:
          Dicionário {vértice: comunidade_id}
      """
      import random
      
      # Inicializar: cada nó com seu próprio label
      labels = {v.index: v.index for v in self.vertices}
      
      for iteration in range(max_iter):
          changed = False
          
          # Processar nós em ordem aleatória (melhora convergência)
          nodes = list(self.adj_list.keys())
          random.shuffle(nodes)
          
          for u in nodes:
              # Pegar vizinhos (ignorando direção)
              neighbors = self.get_all_neighbors(u, ignore_direction=True)
              
              if not neighbors:
                  continue
              
              # Contar labels dos vizinhos
              label_counts = {}
              for neighbor in neighbors:
                  label = labels[neighbor]
                  label_counts[label] = label_counts.get(label, 0) + 1
              
              # Adotar label mais frequente
              most_common_label = max(label_counts, key=label_counts.get)
              
              if labels[u] != most_common_label:
                  labels[u] = most_common_label
                  changed = True
          
          # Convergiu?
          if not changed:
              logger.info(f"Label Propagation convergiu em {iteration + 1} iterações")
              break
      
      return labels

  def calculate_modularity(self, communities: Dict[int, int]) -> float:
      """
      Calcula modularidade da partição em comunidades.
      
      Q = (1/2m) * Σ[Aij - (kikj/2m)] * δ(ci, cj)
      
      Args:
          communities: Dicionário {vértice: comunidade_id}
      
      Returns:
          Modularidade (quanto maior, melhor a divisão).
      """
      m = self.get_edge_count()
      if m == 0:
          return 0.0
      
      # Calcular graus
      degrees = {v.index: self.get_vertex_out_degree(v.index) + self.get_vertex_in_degree(v.index) 
                 for v in self.vertices}
      
      Q = 0.0
      
      for u in self.adj_list:
          for v in self.adj_list:
            if u == v:
                continue
               
            if communities[u] == communities[v]:
                    A_uv = 1.0 if self.has_edge(u, v) else 0.0
                    expected = (degrees[u] * degrees[v]) / (2 * m)
                    Q += A_uv - expected
      
      return Q / (2 * m)

  def calculate_betweenness_centrality(self) -> Dict[int, float]:
      """
      Calcula betweenness centrality para todos os vértices.
      
      Implementação do algoritmo de Brandes (2001).
      Mede quantas vezes um nó aparece nos caminhos mais curtos entre outros nós.
      
      Returns:
          Dicionário {vértice: betweenness}
      """
      betweenness = {v.index: 0.0 for v in self.vertices}
      
      for s in self.adj_list:
          # BFS de s
          stack = []
          predecessors = {v.index: [] for v in self.vertices}
          sigma = {v.index: 0 for v in self.vertices}
          sigma[s] = 1
          distance = {v.index: -1 for v in self.vertices}
          distance[s] = 0
          
          queue = deque([s])
          
          while queue:
              v = queue.popleft()
              stack.append(v)
              
              # Para cada vizinho de v
              for w, _ in self.adj_list[v]:
                  # Caminho via v para w?
                  if distance[w] < 0:
                      queue.append(w)
                      distance[w] = distance[v] + 1
                  
                  # Caminho mais curto para w via v?
                  if distance[w] == distance[v] + 1:
                      sigma[w] += sigma[v]
                      predecessors[w].append(v)
          
          # Acumular dependências
          delta = {v.index: 0.0 for v in self.vertices}
          
          while stack:
              w = stack.pop()
              for v in predecessors[w]:
                  delta[v] += (sigma[v] / sigma[w]) * (1 + delta[w])
              if w != s:
                  betweenness[w] += delta[w]
      
      # Normalizar (para grafo direcionado)
      n = self.vertices_num
      if n > 2:
          factor = 1.0 / ((n - 1) * (n - 2))
          for v in betweenness:
              betweenness[v] *= factor
      
      return betweenness

  def calculate_closeness_centrality(self) -> Dict[int, float]:
      """
      Calcula closeness centrality para todos os vértices.
      
      Mede quão "próximo" um nó está de todos os outros.
      Closeness = (n-1) / Σdistâncias
      
      Returns:
          Dicionário {vértice: closeness}
      """
      closeness = {}
      
      for s in self.adj_list:
          # BFS de s para calcular distâncias
          distance = {v.index: float('inf') for v in self.vertices}
          distance[s] = 0
          
          queue = deque([s])
          
          while queue:
              v = queue.popleft()
              
              for w, _ in self.adj_list[v]:
                  if distance[w] == float('inf'):
                      distance[w] = distance[v] + 1
                      queue.append(w)
          
          # Calcular closeness
          reachable_distances = [d for d in distance.values() if d != float('inf') and d > 0]
          
          if reachable_distances:
              total_distance = sum(reachable_distances)
              closeness[s] = len(reachable_distances) / total_distance
          else:
              closeness[s] = 0.0
      
      return closeness

  def calculate_pagerank(self, alpha: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> Dict[int, float]:
    """
    Calcula PageRank usando Power Iteration.

    Args:
        alpha: Damping factor (probabilidade de seguir links).
        max_iter: Máximo de iterações.
        tol: Tolerância para convergência.

    Returns:
        Dicionário {vértice: pagerank}
    """
    n = self.vertices_num
    if n == 0:
        return {}

    # Inicializar uniformemente
    pagerank = {v.index: 1.0 / n for v in self.vertices}

    for iteration in range(max_iter):
        new_pagerank = {}
        
        for v in self.adj_list:
            # Componente de teleporte
            rank = (1 - alpha) / n
            
            # Componente de links recebidos
            for u in self.adj_list:
                # CORREÇÃO: Pular quando u == v (evita verificar self-loop)
                if u != v:
                    if self.has_edge(u, v):
                        out_degree = self.get_vertex_out_degree(u)
                        if out_degree > 0:
                            rank += alpha * (pagerank[u] / out_degree)
            
            new_pagerank[v] = rank
        
        # Verificar convergência
        diff = sum(abs(new_pagerank[v] - pagerank[v]) for v in pagerank)
        
        pagerank = new_pagerank
        
        if diff < tol:
            logger.info(f"PageRank convergiu em {iteration + 1} iterações")
            break

    return pagerank