# tests/test_adjacency_list.py

import unittest
import tempfile
import os
from pathlib import Path

from graphs_lib.graphs.adjacency_list_graph import AdjacencyListGraph
from graphs_lib.graphs.exceptions import (
InvalidVertexError,
SelfLoopError,
EdgeNotFoundError,
)

class TestAdjacencyListGraph(unittest.TestCase):
    """Testes para implementação com lista de adjacência."""

    def setUp(self):
        self.graph = AdjacencyListGraph(5)

    def test_initialization(self):
        self.assertEqual(self.graph.get_vertex_count(), 5)
        self.assertEqual(self.graph.get_edge_count(), 0)
        self.assertTrue(self.graph.is_empty_graph())

    def test_invalid_vertex_count(self):
        with self.assertRaises(ValueError):
            AdjacencyListGraph(-1)

    def test_add_edge_valid(self):
        self.graph.add_edge(1, 2, 3.5)
        self.assertTrue(self.graph.has_edge(1, 2))
        self.assertEqual(self.graph.get_edge_weight(1, 2), 3.5)
        self.assertEqual(self.graph.get_edge_count(), 1)

    def test_add_edge_idempotent(self):
        self.graph.add_edge(1, 2, 1.0)
        self.graph.add_edge(1, 2, 2.0)
        self.assertEqual(self.graph.get_edge_count(), 1)
        self.assertEqual(self.graph.get_edge_weight(1, 2), 1.0)

    def test_add_edge_self_loop(self):
        with self.assertRaises(SelfLoopError):
            self.graph.add_edge(1, 1)

    def test_add_edge_invalid_vertex(self):
        with self.assertRaises(InvalidVertexError):
            self.graph.add_edge(1, 10)
        with self.assertRaises(InvalidVertexError):
            self.graph.add_edge(0, 2)

    def test_remove_edge(self):
        self.graph.add_edge(1, 2)
        self.assertTrue(self.graph.has_edge(1, 2))
        self.graph.remove_edge(1, 2)
        self.assertFalse(self.graph.has_edge(1, 2))
        self.assertEqual(self.graph.get_edge_count(), 0)

    def test_remove_edge_idempotent(self):
        self.graph.remove_edge(1, 2)
        self.assertEqual(self.graph.get_edge_count(), 0)

    def test_has_edge(self):
        self.assertFalse(self.graph.has_edge(1, 2))
        self.graph.add_edge(1, 2)
        self.assertTrue(self.graph.has_edge(1, 2))
        self.assertFalse(self.graph.has_edge(2, 1))

    def test_edge_weights(self):
        self.graph.add_edge(1, 2, 5.0)
        self.assertEqual(self.graph.get_edge_weight(1, 2), 5.0)
        self.graph.set_edge_weight(1, 2, 10.0)
        self.assertEqual(self.graph.get_edge_weight(1, 2), 10.0)

    def test_get_edge_weight_nonexistent(self):
        with self.assertRaises(EdgeNotFoundError):
            self.graph.get_edge_weight(1, 2)

    def test_set_edge_weight_nonexistent(self):
        with self.assertRaises(EdgeNotFoundError):
            self.graph.set_edge_weight(1, 2, 5.0)

    def test_vertex_degrees(self):
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 3)
        self.graph.add_edge(2, 3)
        self.assertEqual(self.graph.get_vertex_out_degree(1), 2)
        self.assertEqual(self.graph.get_vertex_in_degree(1), 0)
        self.assertEqual(self.graph.get_vertex_out_degree(3), 0)
        self.assertEqual(self.graph.get_vertex_in_degree(3), 2)

    def test_is_successor(self):
        self.graph.add_edge(1, 2)
        self.assertTrue(self.graph.is_successor(2, 1))
        self.assertFalse(self.graph.is_successor(1, 2))

    def test_is_predecessor(self):
        self.graph.add_edge(1, 2)
        self.assertTrue(self.graph.is_predecessor(1, 2))
        self.assertFalse(self.graph.is_predecessor(2, 1))

    def test_is_divergent(self):
        self.graph.add_edge(1, 2)
        self.graph.add_edge(1, 3)
        self.assertTrue(self.graph.is_divergent(1, 2, 1, 3))
        self.assertFalse(self.graph.is_divergent(1, 2, 2, 3))

    def test_is_convergent(self):
        self.graph.add_edge(1, 3)
        self.graph.add_edge(2, 3)
        self.assertTrue(self.graph.is_convergent(1, 3, 2, 3))
        self.assertFalse(self.graph.is_convergent(1, 2, 1, 3))

    def test_is_incident(self):
        self.graph.add_edge(1, 2)
        self.assertTrue(self.graph.is_incident(1, 2, 1))
        self.assertTrue(self.graph.is_incident(1, 2, 2))
        self.assertFalse(self.graph.is_incident(1, 2, 3))

    def test_vertex_weights(self):
        self.graph.set_vertex_weight(1, 7.5)
        self.assertEqual(self.graph.get_vertex_weight(1), 7.5)

    def test_is_empty_graph(self):
        self.assertTrue(self.graph.is_empty_graph())
        self.graph.add_edge(1, 2)
        self.assertFalse(self.graph.is_empty_graph())

    def test_is_complete_graph(self):
        g = AdjacencyListGraph(3)
        self.assertFalse(g.is_complete_graph())
        for i in range(1, 4):
            for j in range(1, 4):
                if i != j:
                    g.add_edge(i, j)
        self.assertTrue(g.is_complete_graph())

    def test_is_connected(self):
        self.graph.add_edge(1, 2)
        self.graph.add_edge(4, 5)
        self.assertFalse(self.graph.is_connected())
        self.graph.add_edge(2, 4)
        self.assertFalse(self.graph.is_connected())
        self.graph.add_edge(3, 1)
        self.assertTrue(self.graph.is_connected())

    def test_export_to_gephi(self):
        self.graph.add_edge(1, 2, 3.5)
        self.graph.add_edge(2, 3, 1.0)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name
        try:
            self.graph.export_to_gephi(temp_path)
            with open(temp_path, "r") as f:
                lines = f.readlines()
                self.assertEqual(lines[0].strip(), "source,target,peso")
                self.assertEqual(len(lines), 3)
        finally:
            os.unlink(temp_path)

    def test_export_to_gephi_with_labels(self):
        self.graph.add_edge(1, 2)
        label_map = {1: "Alice", 2: "Bob"}
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            temp_path = f.name
        try:
            self.graph.export_to_gephi(temp_path, label_map)
            with open(temp_path, "r") as f:
                content = f.read()
                self.assertIn("Alice", content)
                self.assertIn("Bob", content)
        finally:
            os.unlink(temp_path)

class TestAdjacencyListGraphEdgeCases(unittest.TestCase):
    """Testes de edge cases e situações extremas."""

    def test_empty_graph(self):
        g = AdjacencyListGraph(0)
        self.assertEqual(g.get_vertex_count(), 0)
        self.assertTrue(g.is_empty_graph())
        self.assertTrue(g.is_connected())

    def test_single_vertex(self):
        g = AdjacencyListGraph(1)
        self.assertEqual(g.get_vertex_count(), 1)
        self.assertTrue(g.is_empty_graph())
        self.assertTrue(g.is_connected())

    def test_large_weights(self):
        g = AdjacencyListGraph(2)
        large_weight = 1e10
        g.add_edge(1, 2, large_weight)
        self.assertEqual(g.get_edge_weight(1, 2), large_weight)

    def test_zero_weight(self):
        g = AdjacencyListGraph(2)
        g.add_edge(1, 2, 0.0)
        self.assertTrue(g.has_edge(1, 2))
        self.assertEqual(g.get_edge_weight(1, 2), 0.0)

    def test_negative_weight(self):
        g = AdjacencyListGraph(2)
        g.add_edge(1, 2, -5.0)
        self.assertEqual(g.get_edge_weight(1, 2), -5.0)

if __name__ == "__main__":
    unittest.main()