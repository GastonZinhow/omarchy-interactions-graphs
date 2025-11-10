from abc import ABC, abstractmethod

class Vertex:
    def __init__(self, index, label = "", weight = 0):
        self.index = index
        self.label = label
        self.weight = weight

class AbstractGraph(ABC):
    def __init__(self, vertices_num):
        self.vertices_num = vertices_num
        self.vertices = [Vertex(i) for i in range(1, vertices_num + 1)]
        
    def check_vertex(self, u):
        if u < 1 or u > self.vertices_num:
            raise ValueError(f"Vértice {u} inválido.")
        return True
        
    def check_vertices(self, u, v):
        self.check_vertex(u)
        self.check_vertex(v)
        
        if u == v:   
            raise ValueError("Vértices não podem ser iguais.")
        
    def get_vertex_count(self):
        return self.vertices_num
    
    @abstractmethod
    def get_edge_count(self):
        pass
    
    @abstractmethod
    def has_edge(self, u, v):
        pass
    
    @abstractmethod
    def add_edge(self, u, v):
        pass
    
    @abstractmethod
    def remove_edge(self, u, v):
        pass
    
    @abstractmethod
    def is_successor(self, u, v):
        pass
    
    @abstractmethod
    def is_predecessor(self, u, v):
        pass
    
    @abstractmethod
    def is_divergent(self, u1, v1, u2, v2):
        pass
    
    @abstractmethod
    def is_convergent(self, u1, v1, u2, v2):
        pass
    
    @abstractmethod
    def get_vertex_in_degree(self, u):
        pass
    
    @abstractmethod
    def get_vertex_out_degree(self, u):
        pass
    
    def set_vertex_weight(self, u, w):
        self.check_vertex(u)
        
        self.vertices[u].weight = w
    
    def get_vertex_weight(self, u):
        self.check_vertex(u)
        
        return self.vertices[u].weight
    
    @abstractmethod
    def set_edge_weight(self, u, v, w):
        pass
    
    @abstractmethod
    def get_edge_weight(self, u, v):
        pass
    
    # @abstractmethod
    # def is_connected(self):
    #     pass
    
    # @abstractmethod
    # def is_empty_graph(self):
    #     pass
    
    # @abstractmethod
    # def is_complete_graph(self):
    #     pass
