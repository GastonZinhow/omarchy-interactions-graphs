from .AbstractGraph import AbstractGraph

class AdjacencyMatrixGraph(AbstractGraph):
    def __init__(self, vertices_num):
        super().__init__(vertices_num)
        self.mtx = [[float('inf') for _ in range(vertices_num)] for _ in range(vertices_num)]
    
    def print(self):
        mtx_len = len(self.vertices)
        
        print("Matriz de adjancência:\n\n  ", end="")
        for i in range(mtx_len):
            print(f"{self.vertices[i].index} ",end="")
        print()           
        
        for i in range(mtx_len):
            print(f"{self.vertices[i].index} ",end="")
            for j in range(mtx_len):
                print(f"{self.mtx[i][j]} ", end="")
            print()
        print()
        
    def get_edge_count(self):
        mtx_len = len(self.vertices)
        egde_count = 0
        
        for i in range(mtx_len):
            for j in range(mtx_len):
                if self.mtx[i][j] != float('inf'):
                    egde_count += 1
        
        return egde_count

    def has_edge(self, u, v):
        self.check_vertices(u, v)
        
        return self.mtx[u-1][v-1] != float('inf') or self.mtx[v-1][u-1] != float('inf')
    
    def add_edge(self, u, v, w=None):
        self.check_vertices(u, v)
        
        self.mtx[u-1][v-1] = w
        
    def remove_edge(self, u, v):
        self.check_vertices(u, v)
        
        self.mtx[u-1][v-1] = 0
        
    def is_successor(self, u, v):
        self.check_vertices(u, v)
        
        return self.mtx[v-1][u-1] != float('inf')
    
    def is_predecessor(self, u, v):
        self.check_vertices(u, v)
        
        return self.mtx[u-1][v-1] != float('inf')
    
    def is_divergent(self, u1, v1, u2, v2):
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        
        if self.has_edge(u1, v1) and self.has_edge(u2, v2):
            return u1 == u2           
        return False
    
    def is_convergent(self, u1, v1, u2, v2):
        self.check_vertices(u1, v1)
        self.check_vertices(u2, v2)
        
        if self.has_edge(u1, v1) and self.has_edge(u2, v2):
            return v1 == v2
        return False
    
    def get_vertex_in_degree(self, u):
        self.check_vertex(u)
        
        in_degree = 0
        
        for i in range(len(self.vertices)):
            if self.mtx[i][u-1] != float('inf'):
                in_degree += 1
        
        return in_degree
    
    def get_vertex_out_degree(self, u):
        self.check_vertex(u)
        
        out_degree = 0
        
        for i in range(len(self.vertices)):
            if self.mtx[u-1][i] != float('inf'):
                out_degree += 1
        
        return out_degree
        
    def set_edge_weight(self, u, v, w):
        self.check_vertices(u, v)
        
        self.mtx[u-1][v-1] = w
    
    def get_edge_weight(self, u, v):
        self.check_vertices(u, v)
        
        return self.mtx[u-1][v-1]
