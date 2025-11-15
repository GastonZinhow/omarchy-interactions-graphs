from AbstractGraph import AbstractGraph
import csv
from collections import deque


class AdjacencyListGraph(AbstractGraph):
    def __init__(self, vertices_num):
        super().__init__(vertices_num)
        self.adj_list = {v.index: [] for v in self.vertices}

    def print_graph(self):
        print("Lista de adjacência:\n")
        for v in self.vertices:
            edges = ", ".join(f"({dest}, w={w})" for dest, w in self.adj_list[v.index])
            print(f"{v.index}: {edges}")
        print()

    def get_edge_count(self):
        count = sum(len(neighbors) for neighbors in self.adj_list.values())
        return count

    def has_edge(self, u, v):
        self.check_vertices(u, v)
        return any(dest == v for dest, _ in self.adj_list[u])

    def add_edge(self, u, v, w=1):
        self.check_vertices(u, v)
        if u == v:
            raise ValueError("Laço (u==v) não permitido em grafo simples.")
        if not self.has_edge(u, v):
            self.adj_list[u].append((v, w))

    def remove_edge(self, u, v):
        self.check_vertices(u, v)
        self.adj_list[u] = [(dest, w) for dest, w in self.adj_list[u] if dest != v]

    def is_successor(self, u, v):
        self.check_vertices(u, v)
        return any(dest == u for dest, _ in self.adj_list[v])

    def is_predecessor(self, u, v):
        self.check_vertices(u, v)
        return any(dest == v for dest, _ in self.adj_list[u])

    def is_divergent(self, u1, v1, u2, v2):
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        if self.has_edge(u1, v1) and self.has_edge(u2, v2):
            return u1 == u2 and v1 != v2
        return False

    def is_convergent(self, u1, v1, u2, v2):
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        if self.has_edge(u1, v1) and self.has_edge(u2, v2):
            return v1 == v2 and u1 != u2
        return False

    def get_vertex_in_degree(self, u):
        self.check_vertex(u)
        in_degree = 0
        for src in self.adj_list:
            for dest, _ in self.adj_list[src]:
                if dest == u:
                    in_degree += 1
        return in_degree

    def get_vertex_out_degree(self, u):
        self.check_vertex(u)
        return len(self.adj_list[u])

    def set_edge_weight(self, u, v, w):
        self.check_vertices(u, v)
        for i, (dest, _) in enumerate(self.adj_list[u]):
            if dest == v:
                self.adj_list[u][i] = (v, w)
                return
        raise ValueError(f"Aresta ({u}, {v}) não existe.")

    def get_edge_weight(self, u, v):
        self.check_vertices(u, v)
        for dest, w in self.adj_list[u]:
            if dest == v:
                return w
        raise ValueError(f"Aresta ({u}, {v}) não existe.")

    def isIncident(self, u, v, x):
        self.check_vertices(u, v)
        self.check_vertex(x)
        return self.has_edge(u, v) and (x == u or x == v)

    def isEmptyGraph(self):
        return self.get_edge_count() == 0

    def isCompleteGraph(self):
        num_v = self.vertices_num
        for u in self.adj_list:
            vizinhos = set(dest for dest, _ in self.adj_list[u])
            if vizinhos != set(v for v in range(1, num_v + 1) if v != u):
                return False
        return True

    def isConnected(self):
        n = self.vertices_num
        if n == 0:
            return True
        queue = deque([1])
        visited = set([1])
        while queue:
            u = queue.popleft()
            for dest, _ in self.adj_list[u]:
                if dest not in visited:
                    visited.add(dest)
                    queue.append(dest)
            for src in self.adj_list:
                if u != src and any(dest == u for dest, _ in self.adj_list[src]):
                    if src not in visited:
                        visited.add(src)
                        queue.append(src)
        return len(visited) == n

    def exportToGEPHI(self, path, label_map=None):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "peso"])
            for u in self.adj_list:
                for dest, w in self.adj_list[u]:
                    src = label_map[u] if label_map else u
                    tgt = label_map[dest] if label_map else dest
                    writer.writerow([src, tgt, int(w)])
        print(f"[GEPHI] Exportado para: {path}")

    def is_connected(self):
        return self.isConnected()

    def is_empty_graph(self):
        return self.isEmptyGraph()

    def is_complete_graph(self):
        return self.isCompleteGraph()
