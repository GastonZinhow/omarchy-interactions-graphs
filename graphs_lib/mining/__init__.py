"""Funções públicas para mineração de interações GitHub.

Este pacote expõe apenas funções utilitárias; anteriormente fazia referência
à classe ``GitHubCollector`` que não existe no arquivo ``collector.py``.
"""

from .collector import (
	collect_all,
	extract_interactions,
	categorize_edges,
	save_csv,
	save_csv_split,
)

__all__ = [
	"collect_all",
	"extract_interactions",
	"categorize_edges",
	"save_csv",
	"save_csv_split",
]