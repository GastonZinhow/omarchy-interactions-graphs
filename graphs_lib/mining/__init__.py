"""
Módulo de mineração de dados de repositórios.
"""

from .collector import GitHubCollector, save_csv

__all__ = ['GitHubCollector', 'save_csv']