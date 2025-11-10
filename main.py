from graphs_lib.AdjacencyMatrixGraph import AdjacencyMatrixGraph
from graphs_lib.AdjacencyListGraph import AdjacencyListGraph


g = AdjacencyMatrixGraph(5)
# g = AdjacencyListGraph(5)

g.add_edge(1, 3)
g.add_edge(1, 2)
g.add_edge(4, 2)
g.print()
# print("Número de arestas: ", g.get_edge_count())
# print("Número de vertices: ", g.get_vertex_count())
# print("Aresta de 1 pra 3: ", g.has_edge(1, 3))
# print("Aresta de 4 pra 2: ", g.has_edge(4, 2))

# print("Grau de entrada do 1: ", g.get_vertex_in_degree(1))
# print("Grau de entrada do 2: ", g.get_vertex_in_degree(2))

# print("Grau de saída do 1: ", g.get_vertex_out_degree(1))
# print("Grau de saída do 2: ", g.get_vertex_out_degree(2))

# print("4 é predecessor de 2? ", g.is_predecessor(4, 2))
# print("2 é predecessor de 4? ", g.is_predecessor(2, 4))

# print("4 é sucessor de 2? ", g.is_successor(4, 2))
# print("2 é sucessor de 4? ", g.is_successor(2, 4))

# g.remove_edge(1, 3)
# g.print()
# print("Número de arestas: ", g.get_edge_count())
# print("Número de vertices: ", g.get_vertex_count())
# print("Aresta de 1 pra 3: ", g.has_edge(1, 3))
# print("Aresta de 4 pra 2: ", g.has_edge(4, 2))
