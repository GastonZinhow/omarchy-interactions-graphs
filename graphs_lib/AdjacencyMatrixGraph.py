from AbstractGraph import AbstractGraph
import csv


class AdjacencyMatrixGraph(AbstractGraph):
    def __init__(self, vertices_num):
        super().__init__(vertices_num)
        self.mtx = [
            [float("inf") for _ in range(vertices_num)] for _ in range(vertices_num)
        ]

    def print_graph(self):
        mtx_len = len(self.vertices)
        print("\nMatriz de adjacência:\n\n  ", end="")
        for i in range(mtx_len):
            print(f"{self.vertices[i].index} ", end="")
        print()
        for i in range(mtx_len):
            print(f"{self.vertices[i].index} ", end="")
        for j in range(mtx_len):
            v = self.mtx[i][j]
            print("  ." if v == float("inf") else f"{int(v):4d}", end="")
        print()

    print()

    def get_edge_count(self):
        mtx_len = len(self.vertices)
        edge_count = 0
        for i in range(mtx_len):
            for j in range(mtx_len):
                if i != j and self.mtx[i][j] != float("inf"):
                    edge_count += 1
        return edge_count

    def has_edge(self, u, v):
        self.check_vertices(u, v)
        return self.mtx[u - 1][v - 1] != float("inf")

    def add_edge(self, u, v, w=1):
        self.check_vertices(u, v)
        if u == v:
            raise ValueError("Laço (u==v) não permitido em grafo simples.")
        if self.mtx[u - 1][v - 1] == float("inf"):
            self.mtx[u - 1][v - 1] = w

    def remove_edge(self, u, v):
        self.check_vertices(u, v)
        self.mtx[u - 1][v - 1] = float("inf")

    def is_successor(self, u, v):
        self.check_vertices(u, v)
        return self.has_edge(v, u)

    def is_predecessor(self, u, v):
        self.check_vertices(u, v)
        return self.has_edge(u, v)

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
        indeg = 0
        for i in range(self.vertices_num):
            if i != (u - 1) and self.mtx[i][u - 1] != float("inf"):
                indeg += 1
        return indeg

    def get_vertex_out_degree(self, u):
        self.check_vertex(u)
        outdeg = 0
        for j in range(self.vertices_num):
            if j != (u - 1) and self.mtx[u - 1][j] != float("inf"):
                outdeg += 1
        return outdeg

    def set_edge_weight(self, u, v, w):
        self.check_vertices(u, v)
        if self.mtx[u - 1][v - 1] == float("inf"):
            raise ValueError(f"Aresta ({u}, {v}) não existe para atribuir peso.")
        self.mtx[u - 1][v - 1] = w

    def get_edge_weight(self, u, v):
        self.check_vertices(u, v)
        if self.mtx[u - 1][v - 1] == float("inf"):
            raise ValueError(f"Aresta ({u}, {v}) não existe.")
        return self.mtx[u - 1][v - 1]

    def isIncident(self, u, v, x):
        self.check_vertices(u, v)
        self.check_vertex(x)
        return self.has_edge(u, v) and (x == u or x == v)

    def isEmptyGraph(self):
        return self.get_edge_count() == 0

    def isCompleteGraph(self):
        for i in range(self.vertices_num):
            for j in range(self.vertices_num):
                if i != j and self.mtx[i][j] == float("inf"):
                    return False
        return True

    def isConnected(self):
        def bfs(start):
            from collections import deque

            visited = [False] * self.vertices_num
            queue = deque([start])
            visited[start] = True
            while queue:
                u = queue.popleft()
                for v in range(self.vertices_num):
                    if (
                        u != v
                        and (
                            self.mtx[u][v] != float("inf")
                            or self.mtx[v][u] != float("inf")
                        )
                        and not visited[v]
                    ):
                        visited[v] = True
                        queue.append(v)
            return all(visited)

        for i in range(self.vertices_num):
            if bfs(i):
                return True
        return False

    def exportToGEPHI(self, path, label_map=None):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["source", "target", "peso"])
            for i in range(self.vertices_num):
                for j in range(self.vertices_num):
                    if i != j and self.mtx[i][j] != float("inf"):
                        src = label_map[i + 1] if label_map else i + 1
                        tgt = label_map[j + 1] if label_map else j + 1
                        writer.writerow([src, tgt, int(self.mtx[i][j])])
        print(f"[GEPHI] Exportado para: {path}")

    def is_connected(self):
        return self.isConnected()

    def is_empty_graph(self):
        return self.isEmptyGraph()

    def is_complete_graph(self):
        return self.isCompleteGraph()
