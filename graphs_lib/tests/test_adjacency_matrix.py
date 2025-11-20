# tests/test_adjacency_matrix.py

import unittest
import tempfile
import os

from graphs_lib.graphs.adjacency_matrix_graph import AdjacencyMatrixGraph
from graphs_lib.graphs.exceptions import (
InvalidVertexError,
SelfLoopError,
EdgeNotFoundError,
)

class TestAdjacencyMatrixGraph(unittest.TestCase):
    """Testes para implementação com matriz de adjacência."""

    def setUp(self):
        self.graph = AdjacencyMatrixGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.get_vertex_count(), 5)
        self.assertEqual(self.graph.get_edge_count(), 0)
        self.assertTrue(self.graph.is_empty_graph())

    def test_add_edge_valid(self):
        self.graph.add_edge(1, 2, 3.5)
        self.assertTrue(self.graph.has_edge(1, 2))
        self.assertEqual(self.graph.get_edge_weight(1, 2), 3.5)

    def test_add_edge_idempotent(self):
        self.graph.add_edge(1, 2, 1.0)
        self.graph.add_edge(1, 2, 2.0)
        self.assertEqual(self.graph.get_edge_count(), 1)
        self.assertEqual(self.graph.get_edge_weight(1, 2), 1.0)

    def test_add_edge_self_loop(self):
        with self.assertRaises(SelfLoopError):
            self.graph.add_edge(1, 1)

    def test_remove_edge(self):
        self.graph.add_edge(1, 2)
        self.graph.remove_edge(1, 2)
        self.assertFalse(self.graph.has_edge(1, 2))

    def test_vertex_degrees(self):
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 3)
        self.assertEqual(self.graph.get_vertex_out_degree(1), 2)
        self.assertEqual(self.graph.get_vertex_in_degree(2), 1)

    def test_is_complete_graph(self):
        g = AdjacencyMatrixGraph(3)
        for i in range(1, 4):
            for j in range(1, 4):
                if i != j:
                    g.add_edge(i, j)
        self.assertTrue(g.is_complete_graph())

    def test_is_connected(self):
        self.graph.add_edge(1, 2)
        self.graph.add_edge(2, 3)
        self.graph.add_edge(3, 4)
        self.graph.add_edge(4, 5)
        self.assertTrue(self.graph.is_connected())

    def test_export_to_gephi(self):
        self.graph.add_edge(1, 2, 5.0)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name
        try:
            self.graph.export_to_gephi(temp_path)
            with open(temp_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(lines[0].strip(), "source,target,peso")
        finally:
            os.unlink(temp_path)

class TestAdjacencyMatrixGraphEdgeCases(unittest.TestCase):
    """Testes de edge cases para matriz."""

    def test_empty_graph(self):
        g = AdjacencyMatrixGraph(0)
        self.assertEqual(g.get_vertex_count(), 0)
        self.assertTrue(g.is_empty_graph())
        self.assertTrue(g.is_connected())

    def test_single_vertex(self):
        g = AdjacencyMatrixGraph(1)
        self.assertEqual(g.get_vertex_count(), 1)
        self.assertTrue(g.is_empty_graph())
        self.assertTrue(g.is_connected())

    def test_large_weights(self):
        g = AdjacencyMatrixGraph(2)
        large_weight = 1e10
        g.add_edge(1, 2, large_weight)
        self.assertEqual(g.get_edge_weight(1, 2), large_weight)

    def test_zero_weight(self):
        g = AdjacencyMatrixGraph(2)
        g.add_edge(1, 2, 0.0)
        self.assertTrue(g.has_edge(1, 2))
        self.assertEqual(g.get_edge_weight(1, 2), 0.0)

    def test_negative_weight(self):
        g = AdjacencyMatrixGraph(2)
        g.add_edge(1, 2, -5.0)
        self.assertEqual(g.get_edge_weight(1, 2), -5.0)

if __name__ == "__main__":
    unittest.main()