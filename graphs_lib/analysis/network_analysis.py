import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt

# Importar nossa implementação
from ..graphs.adjacency_list_graph import AdjacencyListGraph

logger = logging.getLogger(__name__)

class NetworkAnalyzer:
  """
  Analisador de redes usando APENAS a implementação própria de grafos.
  
  Análise baseada em grafo DIRECIONADO (conforme requisitos do trabalho).
  """

  def __init__(
      self, 
      interactions: List[Tuple[str, str, str, int]], 
      output_dir: Path = Path("output"),
      graph_type: str = "grafo_integrado"
  ):
      """
      Args:
          interactions: Lista de tuplas (source, target, tipo, peso).
          output_dir: Diretório de saída.
          graph_type: Tipo do grafo ('comentarios', 'fechamentos', 'reviews_merges', 'grafo_integrado').
      """
      self.interactions = interactions
      self.output_dir = Path(output_dir)
      self.output_dir.mkdir(parents=True, exist_ok=True)
      self.graph_type = graph_type

      # Mapear usuários para índices
      logger.info("Mapeando usuários para índices...")
      self.user_to_index, self.index_to_user = self._build_user_mapping()
      
      # Construir grafo DIRECIONADO
      logger.info(f"Construindo grafo direcionado ({len(self.user_to_index)} usuários)...")
      self.G = self._build_directed_graph()
      
      logger.info(
          f"Grafo criado: {self.G.get_vertex_count()} nós, "
          f"{self.G.get_edge_count()} arestas"
      )

  # -------------------------
  # Construção do grafo
  # -------------------------
  def _build_user_mapping(self) -> Tuple[Dict[str, int], Dict[int, str]]:
      """
      Cria mapeamento bidirecional entre nomes de usuários e índices.
      
      Returns:
          (user_to_index, index_to_user)
      """
      users = set()
      for src, tgt, _, _ in self.interactions:
          users.add(src)
          users.add(tgt)
      
      users_sorted = sorted(users)
      user_to_index = {user: idx + 1 for idx, user in enumerate(users_sorted)}
      index_to_user = {idx: user for user, idx in user_to_index.items()}
      
      return user_to_index, index_to_user

  def _build_directed_graph(self) -> AdjacencyListGraph:
      """
      Constrói grafo direcionado usando AdjacencyListGraph.
      
      RESPEITA OS PESOS do CSV (2, 3, 4, 5).
      """
      n = len(self.user_to_index)
      G = AdjacencyListGraph(n)
      
      for src, tgt, tipo, peso in self.interactions:
          u = self.user_to_index[src]
          v = self.user_to_index[tgt]
          
          # add_edge agora SOMA pesos se aresta já existe
          G.add_edge(u, v, weight=float(peso))
      
      return G

  # -------------------------
  # Cálculo de centralidades
  # -------------------------
  def _calc_centralities(self) -> pd.DataFrame:
      """
      Calcula TODAS as centralidades usando APENAS métodos próprios.
      
      Returns:
          DataFrame com centralidades.
      """
      logger.info("Calculando centralidades (implementação própria)...")
      
      nodes = list(self.index_to_user.values())
      
      # 1. Graus
      logger.info("  → Calculando graus...")
      in_degree = {}
      out_degree = {}
      degree_total = {}
      
      for idx, user in self.index_to_user.items():
          in_deg = self.G.get_vertex_in_degree(idx)
          out_deg = self.G.get_vertex_out_degree(idx)
          
          in_degree[user] = in_deg
          out_degree[user] = out_deg
          degree_total[user] = in_deg + out_deg
      
      # 2. Betweenness (Algoritmo de Brandes)
      logger.info("  → Calculando Betweenness Centrality (Brandes)...")
      betweenness_dict = self.G.calculate_betweenness_centrality()
      betweenness = {self.index_to_user[idx]: val for idx, val in betweenness_dict.items()}
      
      # 3. Closeness
      logger.info("  → Calculando Closeness Centrality...")
      closeness_dict = self.G.calculate_closeness_centrality()
      closeness = {self.index_to_user[idx]: val for idx, val in closeness_dict.items()}
      
      # 4. PageRank
      logger.info("  → Calculando PageRank...")
      pagerank_dict = self.G.calculate_pagerank()
      pagerank = {self.index_to_user[idx]: val for idx, val in pagerank_dict.items()}
      
      # Montar DataFrame
      df = pd.DataFrame({
          'colaborador': nodes,
          'in_degree': [in_degree[n] for n in nodes],
          'out_degree': [out_degree[n] for n in nodes],
          'degree_total': [degree_total[n] for n in nodes],
          'betweenness': [betweenness[n] for n in nodes],
          'closeness': [closeness[n] for n in nodes],
          'pagerank': [pagerank[n] for n in nodes],
      })
      
      df = df.sort_values('pagerank', ascending=False)
      return df

  # -------------------------
  # Métricas de estrutura
  # -------------------------
  def _calc_network_metrics(self) -> Dict[str, Any]:
      """
      Calcula métricas globais de estrutura usando implementação própria.
      """
      logger.info("Calculando métricas estruturais (implementação própria)...")
      
      # 1. Densidade
      logger.info("  → Densidade...")
      densidade = self.G.calculate_density()
      
      # 2. Clustering Coefficient médio
      logger.info("  → Clustering Coefficient...")
      clustering_media = self.G.calculate_average_clustering_coefficient()
      
      # Clustering local por nó
      clustering_dict = {}
      for idx, user in self.index_to_user.items():
          clustering_dict[user] = self.G.calculate_clustering_coefficient(idx)
      
      # 3. Assortatividade
      logger.info("  → Assortatividade...")
      assortatividade = self.G.calculate_degree_assortativity()
      
      # 4. Detecção de comunidades
      logger.info("  → Detecção de comunidades (Label Propagation)...")
      comunidades_idx = self.G.detect_communities_label_propagation()
      
      # Mapear de volta para nomes
      comu_por_no = {self.index_to_user[idx]: comu 
                     for idx, comu in comunidades_idx.items()}
      
      # Agrupar comunidades
      comunidades_grupos = {}
      for user, comu_id in comu_por_no.items():
          if comu_id not in comunidades_grupos:
              comunidades_grupos[comu_id] = []
          comunidades_grupos[comu_id].append(user)
      
      comunidades = list(comunidades_grupos.values())
      
      # 5. Modularidade
      logger.info("  → Modularidade...")
      modularidade = self.G.calculate_modularity(comunidades_idx)
      
      return {
          "densidade": densidade,
          "clustering_media": clustering_media,
          "assortatividade": assortatividade,
          "modularidade": modularidade,
          "n_comunidades": len(comunidades),
          "comunidades": comunidades,
          "clustering_dict": clustering_dict,
          "comu_por_no": comu_por_no,
      }

  def _export_network_metrics(self, metrics: Dict[str, Any]):
      """Salva métricas globais e comunidades em CSV."""
      path_globais = self.output_dir / "metricas_estruturais.csv"
      path_comunidades = self.output_dir / "comunidades.csv"

      globals_df = pd.DataFrame({
          "metrica": [
              "densidade",
              "clustering_media",
              "assortatividade",
              "modularidade",
              "n_comunidades",
          ],
          "valor": [
              metrics["densidade"],
              metrics["clustering_media"],
              metrics["assortatividade"],
              metrics["modularidade"],
              metrics["n_comunidades"],
          ],
      })
      
      try:
          globals_df.to_csv(path_globais, index=False, encoding="utf-8")
          logger.info(f"💾 Exportado: {path_globais}")
      except Exception as e:
          logger.error(f"Erro salvando metricas_estruturais.csv: {e}")

      # Comunidades
      comu_por_no = metrics["comu_por_no"]
      linhas_comu = [
          {"colaborador": user, "grupo_comunidade": comu}
          for user, comu in comu_por_no.items()
      ]
      
      if linhas_comu:
          df_comu = pd.DataFrame(linhas_comu)
          try:
              df_comu.to_csv(path_comunidades, index=False, encoding="utf-8")
              logger.info(f"💾 Exportado: {path_comunidades}")
          except Exception as e:
              logger.error(f"Erro salvando comunidades.csv: {e}")

  def _merge_structural_metrics_to_df(
      self, df: pd.DataFrame, metrics: Dict[str, Any]
  ) -> pd.DataFrame:
      """Adiciona clustering local e comunidade ao DataFrame."""
      df["clustering_local"] = df["colaborador"].map(metrics["clustering_dict"])
      df["grupo_comunidade"] = df["colaborador"].map(metrics["comu_por_no"])
      return df

  # -------------------------
  # Bridging Ties
  # -------------------------
  def get_top_bridging_ties(
      self, df: pd.DataFrame, metrics: Dict[str, Any], n=5
  ) -> pd.DataFrame:
      """Identifica top N bridging ties (maior betweenness)."""
      top = df.nlargest(n, 'betweenness')[
          ['colaborador', 'betweenness', 'degree_total', 'pagerank', 'grupo_comunidade']
      ].copy()
      
      comu_por_no = metrics.get('comu_por_no', {})
      comunidades_por_usuario = {}
      
      for idx, row in top.iterrows():
          user = row['colaborador']
          user_idx = self.user_to_index[user]
          
          comunidades_vizinhas = set()
          
          # Pegar vizinhos (sucessores e predecessores)
          vizinhos = self.G.get_all_neighbors(user_idx, ignore_direction=True)
          
          for viz_idx in vizinhos:
              viz_user = self.index_to_user[viz_idx]
              if viz_user in comu_por_no:
                  comunidades_vizinhas.add(comu_por_no[viz_user])
          
          # Adicionar própria comunidade
          if user in comu_por_no:
              comunidades_vizinhas.add(comu_por_no[user])
          
          comunidades_por_usuario[user] = len(comunidades_vizinhas)
      
      top['comunidades_conectadas'] = top['colaborador'].map(comunidades_por_usuario)
      top = top.sort_values('betweenness', ascending=False)
      
      logger.info(f"Top {n} bridging ties identificados:")
      for idx, row in top.iterrows():
          logger.info(
              f"  {row['colaborador']}: betweenness={row['betweenness']:.4f}, "
              f"conecta {row['comunidades_conectadas']} comunidades"
          )
      
      return top

  # -------------------------
  # Exportações
  # -------------------------
  def _export_gephi(self):
      """Exporta grafo para GEXF (usando NetworkX apenas para exportação)."""
      import networkx as nx
      
      path_gephi = self.output_dir / f"grafo_{self.graph_type}.gexf"
      
      # Converter para NetworkX apenas para exportação
      G_nx = nx.DiGraph()
      
      for idx, user in self.index_to_user.items():
          G_nx.add_node(user, label=user)
      
      for u_idx in range(1, self.G.get_vertex_count() + 1):
          for v_idx, weight in self.G.adj_list[u_idx]:
              u_user = self.index_to_user[u_idx]
              v_user = self.index_to_user[v_idx]
              G_nx.add_edge(u_user, v_user, weight=weight)
      
      try:
          nx.write_gexf(G_nx, str(path_gephi))
          logger.info(f"💾 Exportado: {path_gephi}")
      except Exception as e:
          logger.error(f"Erro exportando GEXF: {e}")

  def _export_centralities(self, df: pd.DataFrame):
      """Exporta centralidades para CSV."""
      path = self.output_dir / f"centralidades_{self.graph_type}.csv"
      try:
          df.to_csv(path, index=False, encoding="utf-8")
          logger.info(f"💾 Exportado: {path}")
      except Exception as e:
          logger.error(f"Erro salvando centralidades: {e}")

  # -------------------------
  # Visualizações
  # -------------------------
  def _plot_metrics_panel(self, df: pd.DataFrame):
      """Gera painel de gráficos de análise."""
      logger.info(f"Gerando gráfico de análise...")
      
      fig, axes = plt.subplots(2, 3, figsize=(18, 10))
      fig.suptitle(f"Análise - Grafo {self.graph_type.upper()}", fontsize=16, fontweight="bold")

      # 1 - Top 20 PageRank
      top_pr = df.nlargest(20, "pagerank")
      axes[0, 0].barh(range(len(top_pr)), top_pr["pagerank"])
      axes[0, 0].set_yticks(range(len(top_pr)))
      axes[0, 0].set_yticklabels(top_pr["colaborador"], fontsize=8)
      axes[0, 0].invert_yaxis()
      axes[0, 0].set_title("Top 20 - PageRank")

      # 2 - Top 20 Betweenness
      top_b = df.nlargest(20, "betweenness")
      axes[0, 1].barh(range(len(top_b)), top_b["betweenness"])
      axes[0, 1].set_yticks(range(len(top_b)))
      axes[0, 1].set_yticklabels(top_b["colaborador"], fontsize=8)
      axes[0, 1].invert_yaxis()
      axes[0, 1].set_title("Top 20 - Betweenness")

      # 3 - Distribuição de Grau
      axes[0, 2].hist(df["degree_total"].dropna(), bins=50)
      axes[0, 2].set_title("Distribuição de Grau Total")
      axes[0, 2].set_xlabel("Grau")
      axes[0, 2].set_yscale("log")

      # 4 - Degree vs PageRank
      axes[1, 0].scatter(df["degree_total"], df["pagerank"], s=30)
      axes[1, 0].set_xlabel("Degree Total")
      axes[1, 0].set_ylabel("PageRank")
      axes[1, 0].set_title("Degree vs PageRank")

      # 5 - Betweenness vs Closeness
      axes[1, 1].scatter(df["betweenness"], df["closeness"], s=30)
      axes[1, 1].set_xlabel("Betweenness")
      axes[1, 1].set_ylabel("Closeness")
      axes[1, 1].set_title("Betweenness vs Closeness")

      # 6 - Boxplot das centralidades
      cols_for_box = [
          df["degree_total"].dropna().values,
          df["betweenness"].dropna().values,
          df["closeness"].dropna().values,
          df["pagerank"].dropna().values,
      ]
      axes[1, 2].boxplot(
          cols_for_box,
          labels=["degree", "betweenness", "closeness", "pagerank"],
          notch=True,
      )
      axes[1, 2].set_title("Distribuição das Centralidades")

      plt.tight_layout(rect=[0, 0.03, 1, 0.95])
      
      out = self.output_dir / f"analise_grafo_{self.graph_type}.png"
      try:
          plt.savefig(out, dpi=200, bbox_inches="tight")
          plt.close(fig)
          logger.info(f"💾 Gerado: {out}")
      except Exception as e:
          logger.error(f"Erro salvando gráfico: {e}")

  # -------------------------
  # Análise completa
  # -------------------------
  def executar_analise_completa(self) -> Dict[str, Any]:
      """Executa análise completa do grafo."""
      
      # 1. Calcular centralidades
      df = self._calc_centralities()

      # 2. Calcular métricas estruturais
      metrics = self._calc_network_metrics()
      self._export_network_metrics(metrics)
      df = self._merge_structural_metrics_to_df(df, metrics)

      # 3. Identificar bridging ties
      bridging_ties = self.get_top_bridging_ties(df, metrics, n=5)
      path_bridging = self.output_dir / "bridging_ties.csv"
      try:
          bridging_ties.to_csv(path_bridging, index=False, encoding="utf-8")
          logger.info(f"💾 Exportado: {path_bridging}")
      except Exception as e:
          logger.error(f"Erro salvando bridging_ties.csv: {e}")

      # 4. Exportações
      self._export_gephi()
      self._export_centralities(df)
      self._plot_metrics_panel(df)

      resultados = {
          "tipo_grafo": self.graph_type,
          "quantidade_nos": self.G.get_vertex_count(),
          "quantidade_arestas": self.G.get_edge_count(),
          "densidade": metrics["densidade"],
          "clustering_media": metrics["clustering_media"],
          "assortatividade": metrics["assortatividade"],
          "modularidade": metrics["modularidade"],
          "n_comunidades": metrics["n_comunidades"],
          "arquivo_grafo_gexf": str(self.output_dir / f"grafo_{self.graph_type}.gexf"),
          "arquivo_centralidades": str(self.output_dir / f"centralidades_{self.graph_type}.csv"),
          "arquivo_analise": str(self.output_dir / f"analise_grafo_{self.graph_type}.png"),
          "arquivo_metricas": str(self.output_dir / "metricas_estruturais.csv"),
          "arquivo_comunidades": str(self.output_dir / "comunidades.csv"),
          "arquivo_bridging_ties": str(path_bridging),
      }

      logger.info(f"Análise completa do grafo '{self.graph_type}' executada.")
      return resultados