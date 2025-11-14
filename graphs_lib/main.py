import csv
from AdjacencyMatrixGraph import AdjacencyMatrixGraph

# LEITURA do interacoes.csv
users = set()
edges = []
with open("interacoes.csv", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        users.add(row["source"])
        users.add(row["target"])
        edges.append((row["source"], row["target"], float(row["peso"])))

users = sorted(users)
name2idx = {name: idx + 1 for idx, name in enumerate(users)}
idx2name = {idx + 1: name for idx, name in enumerate(users)}

g = AdjacencyMatrixGraph(len(users))

for src, tgt, peso in edges:
    i = name2idx[src]
    j = name2idx[tgt]
    if i != j:
        if g.has_edge(i, j):
            old = g.get_edge_weight(i, j)
            g.set_edge_weight(i, j, old + peso)
        else:
            g.add_edge(i, j, peso)

# Testes
g.print_graph()
print("Total de vértices:", g.get_vertex_count())
print("Total de arestas:", g.get_edge_count())
print("Grafo vazio?", g.isEmptyGraph())
print("Grafo completo?", g.isCompleteGraph())
print("Grafo conectado?", g.isConnected())

g.exportToGEPHI("grafo_exportado.csv", label_map=idx2name)
print("Exportado para Gephi com sucesso!")
