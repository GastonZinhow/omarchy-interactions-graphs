import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class NetworkAnalyzer:

    def __init__(
        self, interactions: List[Tuple[str, str]], output_dir: Path = Path("output")
    ):
        self.interactions = interactions
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # construir grafos
        logger.info("Construindo grafos (direcionado e não-direcionado)...")
        self.G_dir = self._build_directed_graph()
        self.G_und = nx.Graph(self.G_dir)  # sumariza múltiplas arestas automaticamente
        logger.info(
            f"Grafo direcionado: {self.G_dir.number_of_nodes()} nós, {self.G_dir.number_of_edges()} arestas"
        )
        logger.info(
            f"Grafo não-direcionado: {self.G_und.number_of_nodes()} nós, {self.G_und.number_of_edges()} arestas"
        )

    # -------------------------
    # Construção dos grafos
    # -------------------------
    def _build_directed_graph(self) -> nx.DiGraph:
        G = nx.DiGraph()
        for src, tgt in self.interactions:
            if G.has_edge(src, tgt):
                # soma peso
                cur_w = G[src][tgt].get("weight", 1)
                G[src][tgt]["weight"] = cur_w + 1
            else:
                G.add_edge(src, tgt, weight=1)
        return G

    # -------------------------
    # PageRank puro (NumPy)
    # -------------------------
    def _pagerank_numpy(
        self, G: nx.Graph, alpha=0.85, max_iter=100, tol=1e-6
    ) -> Dict[str, float]:
        nodes = list(G.nodes())
        n = len(nodes)
        if n == 0:
            return {}

        # matriz de adjacência orientada para PageRank: coluna estocástica
        A = nx.to_numpy_array(G, nodelist=nodes, weight="weight", dtype=float)
        M = A.T.copy()
        col_sum = M.sum(axis=0)
        col_sum[col_sum == 0] = 1.0
        M = M / col_sum

        r = np.ones(n) / n
        teleport = np.ones(n) / n

        for it in range(max_iter):
            r_new = alpha * (M @ r) + (1 - alpha) * teleport
            if np.linalg.norm(r_new - r, 1) < tol:
                r = r_new
                break
            r = r_new
        return {nodes[i]: float(r[i]) for i in range(n)}

    # -------------------------
    # Cálculo de centralidades (normal)
    # -------------------------
    def _calc_centralities_undirected(self) -> pd.DataFrame:
        G = self.G_und
        nodes = list(G.nodes())

        degree = dict(G.degree(weight=None))
        degree_weighted = dict(G.degree(weight="weight"))
        degree_centrality = nx.degree_centrality(G)
        betweenness = nx.betweenness_centrality(G, weight="weight")
        closeness = nx.closeness_centrality(G, distance="weight")
        try:
            eigen = nx.eigenvector_centrality_numpy(G, weight="weight")
        except Exception:
            eigen = {n: 0.0 for n in nodes}
        pagerank = self._pagerank_numpy(G)

        df = pd.DataFrame(
            {
                "colaborador": nodes,
                "degree": [degree.get(n, 0) for n in nodes],
                "degree_weighted": [degree_weighted.get(n, 0) for n in nodes],
                "degree_centrality": [degree_centrality.get(n, 0) for n in nodes],
                "betweenness": [betweenness.get(n, 0) for n in nodes],
                "closeness": [closeness.get(n, 0) for n in nodes],
                "eigenvector": [eigen.get(n, 0) for n in nodes],
                "pagerank": [pagerank.get(n, 0) for n in nodes],
            }
        )
        df = df.sort_values("pagerank", ascending=False)
        return df

    # -------------------------
    # Centralidade no dirigido
    # -------------------------
    def _calc_centralities_directed(self) -> pd.DataFrame:
        G = self.G_dir
        nodes = list(G.nodes())

        in_deg = dict(G.in_degree(weight=None))
        out_deg = dict(G.out_degree(weight=None))
        in_deg_w = dict(G.in_degree(weight="weight"))
        out_deg_w = dict(G.out_degree(weight="weight"))

        try:
            betweenness = nx.betweenness_centrality(G, weight="weight")
        except Exception:
            betweenness = {n: 0.0 for n in nodes}
        try:
            closeness = nx.closeness_centrality(G, distance="weight")
        except Exception:
            closeness = {n: 0.0 for n in nodes}

        pagerank = self._pagerank_numpy(G)

        df = pd.DataFrame(
            {
                "colaborador": nodes,
                "in_degree": [in_deg.get(n, 0) for n in nodes],
                "out_degree": [out_deg.get(n, 0) for n in nodes],
                "in_degree_weighted": [in_deg_w.get(n, 0) for n in nodes],
                "out_degree_weighted": [out_deg_w.get(n, 0) for n in nodes],
                "betweenness": [betweenness.get(n, 0) for n in nodes],
                "closeness": [closeness.get(n, 0) for n in nodes],
                "pagerank": [pagerank.get(n, 0) for n in nodes],
            }
        )
        df = df.sort_values("pagerank", ascending=False)
        return df

    # -------------------------
    # BRIDGING TIES (TOP N BETWEENNESS)
    # -------------------------
    def get_top_bridging_ties(self, df_und: pd.DataFrame, metrics: Dict[str, Any], n=5) -> pd.DataFrame:
        # Pegar top N por betweenness
        top = df_und.nlargest(n, 'betweenness')[
            ['colaborador', 'betweenness', 'degree', 'pagerank', 'grupo_comunidade']
        ].copy()
        
        # Mapear comunidades por nó
        comu_por_no = metrics.get('comu_por_no', {})
        
        # Contar quantas comunidades diferentes cada usuário conecta
        comunidades_por_usuario = {}
        
        for idx, row in top.iterrows():
            user = row['colaborador']
            comunidades_vizinhas = set()
            
            # Verificar se o usuário existe no grafo
            if self.G_und.has_node(user):
                # Para cada vizinho, pegar sua comunidade
                for neighbor in self.G_und.neighbors(user):
                    if neighbor in comu_por_no:
                        comu_vizinho = comu_por_no[neighbor]
                        if comu_vizinho is not None:
                            comunidades_vizinhas.add(comu_vizinho)
                
                # Adicionar a própria comunidade do usuário
                if user in comu_por_no and comu_por_no[user] is not None:
                    comunidades_vizinhas.add(comu_por_no[user])
            
            comunidades_por_usuario[user] = len(comunidades_vizinhas)
        
        # Adicionar coluna de comunidades conectadas
        top['comunidades_conectadas'] = top['colaborador'].map(comunidades_por_usuario)
        
        # Ordenar por betweenness decrescente
        top = top.sort_values('betweenness', ascending=False)
        
        logger.info(f"Top {n} bridging ties identificados:")
        for idx, row in top.iterrows():
            logger.info(
                f"  {row['colaborador']}: betweenness={row['betweenness']:.4f}, "
                f"conecta {row['comunidades_conectadas']} comunidades"
            )
        
        return top
    
    def export_bridging_ties_subgraph(self, bridging_ties: pd.DataFrame, metrics: Dict[str, Any]) -> None:
        bridges_list = bridging_ties['colaborador'].tolist()

        # Criar subgrafo com bridges + vizinhos de 1° grau
        nodes_to_include = set(bridges_list)

        # Adicionar todos os vizinhos diretos dos bridges
        for bridge in bridges_list:
            if self.G_und.has_node(bridge):
                neighbors = list(self.G_und.neighbors(bridge))
                nodes_to_include.update(neighbors)

        # Extrair subgrafo
        subgraph = self.G_und.subgraph(nodes_to_include).copy()

        logger.info(f"Subgrafo de bridging ties: {subgraph.number_of_nodes()} nós, {subgraph.number_of_edges()} arestas")

        # Adicionar atributos aos nós para visualização no Gephi
        comu_por_no = metrics.get('comu_por_no', {})

        for node in subgraph.nodes():
            # Marcar se é um bridge
            is_bridge = node in bridges_list
            subgraph.nodes[node]['is_bridge'] = 'Sim' if is_bridge else 'Não'
            
            # Adicionar comunidade
            subgraph.nodes[node]['comunidade'] = comu_por_no.get(node, 0)
            
            # Tamanho sugerido (bridges maiores)
            if is_bridge:
                subgraph.nodes[node]['size'] = 150.0
            else:
                subgraph.nodes[node]['size'] = 20.0
            
            # Label
            subgraph.nodes[node]['label'] = str(node)

        # Exportar
        path_bridges_gexf = self.output_dir / "grafo_bridging_ties.gexf"
        try:
            nx.write_gexf(subgraph, str(path_bridges_gexf))
            logger.info(f"💾 Subgrafo de bridging ties exportado: {path_bridges_gexf}")
            print(f"✓ Subgrafo exportado: {path_bridges_gexf}")
            print(f"  → {subgraph.number_of_nodes()} nós (5 bridges + {subgraph.number_of_nodes()-5} vizinhos)")
            print(f"  → {subgraph.number_of_edges()} arestas")
        except Exception as e:
            logger.error(f"Erro exportando grafo_bridging_ties.gexf: {e}")

    # -------------------------
    # MÉTRICAS DE ESTRUTURA/COESÃO/COMUNIDADE
    # -------------------------
    def _calc_network_metrics(self, G: nx.Graph) -> Dict[str, Any]:
        """
        Calcula métricas globais de estrutura e comunidade do grafo.
        """
        # Densidade
        densidade = nx.density(G)

        # Clustering coeficiente local (dict) e médio (float)
        clustering_dict = nx.clustering(G, weight="weight")
        clustering_media = (
            float(np.mean(list(clustering_dict.values()))) if clustering_dict else 0.0
        )

        # Assortatividade (correlação entre graus)
        try:
            assortatividade = nx.degree_assortativity_coefficient(G)
        except Exception:
            assortatividade = None

        # Detecção de comunidades e modularidade
        try:
            from networkx.algorithms.community import greedy_modularity_communities

            comunidades = list(greedy_modularity_communities(G))
            modularidade = nx.algorithms.community.modularity(G, comunidades)
            comu_por_no = {}
            for i, grupo in enumerate(comunidades):
                for n in grupo:
                    comu_por_no[n] = i + 1
        except Exception as e:
            logger.warning(f"Erro na detecção de comunidades/modularidade: {e}")
            comunidades = []
            modularidade = None
            comu_por_no = {}

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
        """
        Salva as métricas globais e comunidades em CSV.
        """
        path_globais = self.output_dir / "metricas_estruturais.csv"
        path_comunidades = self.output_dir / "comunidades.csv"

        globals_df = pd.DataFrame(
            {
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
            }
        )
        try:
            globals_df.to_csv(path_globais, index=False, encoding="utf-8")
            logger.info(f"💾 Exportado: {path_globais}")
        except Exception as e:
            logger.error(f"Erro salvando metricas_estruturais.csv: {e}")

        comunidades = metrics["comunidades"]
        comu_por_no = metrics["comu_por_no"]
        linhas_comu = []
        for i, grupo in enumerate(comunidades):
            for n in grupo:
                linhas_comu.append({"colaborador": n, "grupo_comunidade": i + 1})
        if linhas_comu:
            df_comu = pd.DataFrame(linhas_comu)
            try:
                df_comu.to_csv(path_comunidades, index=False, encoding="utf-8")
                logger.info(f"💾 Exportado: {path_comunidades}")
            except Exception as e:
                logger.error(f"Erro salvando comunidades.csv: {e}")

    def _merge_structural_metrics_to_df(
        self, df: pd.DataFrame, metrics: Dict[str, Any]
    ):
        """
        Adiciona ao DF:
        - clustering_local (coeficiente por nó)
        - grupo_comunidade (label numérico da comunidade)
        """
        df["clustering_local"] = df["colaborador"].map(metrics["clustering_dict"])
        if "comu_por_no" in metrics and metrics["comu_por_no"]:
            df["grupo_comunidade"] = df["colaborador"].map(metrics["comu_por_no"])
        else:
            df["grupo_comunidade"] = None
        return df

    # -------------------------
    # Exportações (GEXF / CSV)
    # -------------------------
    def _export_gexf(self):
        path_normal = self.output_dir / "grafo_normal.gexf"
        path_inter = self.output_dir / "grafo_interacoes.gexf"
        try:
            nx.write_gexf(self.G_und, str(path_normal))
            logger.info(f"💾 Exportado: {path_normal}")
        except Exception as e:
            logger.error(f"Erro exportando grafo_normal.gexf: {e}")

        try:
            nx.write_gexf(self.G_dir, str(path_inter))
            logger.info(f"💾 Exportado: {path_inter}")
        except Exception as e:
            logger.error(f"Erro exportando grafo_interacoes.gexf: {e}")

    def _export_centralities(self, df_und: pd.DataFrame, df_dir: pd.DataFrame):
        path1 = self.output_dir / "centralidades_normal.csv"
        path2 = self.output_dir / "centralidades_interacoes.csv"
        try:
            df_und.to_csv(path1, index=False, encoding="utf-8")
            logger.info(f"💾 Exportado: {path1}")
        except Exception as e:
            logger.error(f"Erro salvando centralidades_normal.csv: {e}")

        try:
            df_dir.to_csv(path2, index=False, encoding="utf-8")
            logger.info(f"💾 Exportado: {path2}")
        except Exception as e:
            logger.error(f"Erro salvando centralidades_interacoes.csv: {e}")

    # -------------------------
    # Visualizações (Matplotlib)
    # -------------------------
    def _plot_metrics_panel(self, df: pd.DataFrame, titulo: str, output_name: str):
        logger.info(f"Gerando gráfico: {output_name}")
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(titulo, fontsize=16, fontweight="bold")

        # 1 - Top 20 PageRank
        top_pr = df.nlargest(20, "pagerank")
        axes[0, 0].barh(range(len(top_pr)), top_pr["pagerank"])
        axes[0, 0].set_yticks(range(len(top_pr)))
        axes[0, 0].set_yticklabels(top_pr["colaborador"], fontsize=8)
        axes[0, 0].invert_yaxis()
        axes[0, 0].set_title("Top 20 - PageRank")

        # 2 - Top 20 Betweenness
        if "betweenness" in df.columns:
            top_b = df.nlargest(20, "betweenness")
            axes[0, 1].barh(range(len(top_b)), top_b["betweenness"])
            axes[0, 1].set_yticks(range(len(top_b)))
            axes[0, 1].set_yticklabels(top_b["colaborador"], fontsize=8)
            axes[0, 1].invert_yaxis()
            axes[0, 1].set_title("Top 20 - Betweenness")
        else:
            axes[0, 1].text(0.5, 0.5, "No betweenness", ha="center")
            axes[0, 1].set_title("Betweenness")

        # 3 - Distribuição de Grau
        graus = df["degree"] if "degree" in df.columns else df["in_degree"]
        axes[0, 2].hist(graus.dropna(), bins=min(50, int(max(graus.dropna()) + 1)))
        axes[0, 2].set_title("Distribuição de Grau")
        axes[0, 2].set_xlabel("Grau")
        axes[0, 2].set_yscale("log")

        # 4 - Degree vs PageRank
        axes[1, 0].scatter(
            df.iloc[
                :,
                (
                    df.columns.get_loc("degree")
                    if "degree" in df.columns
                    else df.columns.get_loc("in_degree")
                ),
            ],
            df["pagerank"],
            s=30,
        )
        axes[1, 0].set_xlabel("Degree")
        axes[1, 0].set_ylabel("PageRank")
        axes[1, 0].set_title("Degree vs PageRank")

        # 5 - Betweenness vs Closeness
        if "betweenness" in df.columns and "closeness" in df.columns:
            axes[1, 1].scatter(df["betweenness"], df["closeness"], s=30)
            axes[1, 1].set_xlabel("Betweenness")
            axes[1, 1].set_ylabel("Closeness")
            axes[1, 1].set_title("Betweenness vs Closeness")
        else:
            axes[1, 1].text(0.5, 0.5, "No data", ha="center")
            axes[1, 1].set_title("Betweenness vs Closeness")

        # 6 - Boxplot das centralidades
        cols_for_box = []
        for c in ("degree", "betweenness", "closeness", "pagerank"):
            if c in df.columns:
                cols_for_box.append(df[c].dropna().values)
        if cols_for_box:
            axes[1, 2].boxplot(
                cols_for_box,
                labels=[
                    c
                    for c in ("degree", "betweenness", "closeness", "pagerank")
                    if c in df.columns
                ],
                notch=True,
            )
            axes[1, 2].set_title("Distribuição das Centralidades")
        else:
            axes[1, 2].text(0.5, 0.5, "No data", ha="center")
            axes[1, 2].set_title("Centralidades")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out = self.output_dir / output_name
        try:
            plt.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            logger.info(f"💾 Gerado: {out}")
        except Exception as e:
            logger.error(f"Erro salvando {out}: {e}")


    def executar_analise_completa(self) -> Dict[str, Any]:
        # calcular centralidades
        df_und = self._calc_centralities_undirected()
        df_dir = self._calc_centralities_directed()

        # Calcular métricas estruturais/globais/comunidade
        metrics = self._calc_network_metrics(self.G_und)
        self._export_network_metrics(metrics)
        df_und = self._merge_structural_metrics_to_df(df_und, metrics)

        # Identificar e exportar bridging ties (top 5 betweenness)
        bridging_ties = self.get_top_bridging_ties(df_und, metrics, n=5)
        path_bridging = self.output_dir / "bridging_ties.csv"
        try:
            bridging_ties.to_csv(path_bridging, index=False, encoding="utf-8")
            logger.info(f"💾 Exportado: {path_bridging}")
        except Exception as e:
            logger.error(f"Erro salvando bridging_ties.csv: {e}")
        self.export_bridging_ties_subgraph(bridging_ties, metrics)

        # exportar GEXF
        self._export_gexf()

        # exportar CSVs
        self._export_centralities(df_und, df_dir)

        # produzir gráficos
        self._plot_metrics_panel(
            df_und,
            "Análise - Grafo Normal (não-direcionado)",
            "analise_grafo_normal.png",
        )
        self._plot_metrics_panel(
            df_dir,
            "Análise - Grafo de Interações (direcionado)",
            "analise_grafo_interacoes.png",
        )

        resultados = {
            "quantidade_nos": self.G_dir.number_of_nodes(),
            "quantidade_arestas": self.G_dir.number_of_edges(),
            "componentes_fortes": nx.number_strongly_connected_components(self.G_dir),
            "componentes_fracos": nx.number_weakly_connected_components(self.G_dir),
            "arquivo_grafo_normal": str(self.output_dir / "grafo_normal.gexf"),
            "arquivo_grafo_interacoes": str(self.output_dir / "grafo_interacoes.gexf"),
            "arquivo_grafo_bridging_ties": str(self.output_dir / "grafo_bridging_ties.gexf"), 
            "arquivo_analise_grafo_normal": str(self.output_dir / "analise_grafo_normal.png"),
            "arquivo_analise_grafo_interacoes": str(self.output_dir / "analise_grafo_interacoes.png"),
            "arquivo_centralidades_normal": str(self.output_dir / "centralidades_normal.csv"),
            "arquivo_centralidades_interacoes": str(self.output_dir / "centralidades_interacoes.csv"),
            "arquivo_metricas_estruturais": str(self.output_dir / "metricas_estruturais.csv"),
            "arquivo_comunidades": str(self.output_dir / "comunidades.csv"),
            "arquivo_bridging_ties": str(path_bridging),
            "densidade": metrics["densidade"],
            "clustering_media": metrics["clustering_media"],
            "assortatividade": metrics["assortatividade"],
            "modularidade": metrics["modularidade"],
            "n_comunidades": metrics["n_comunidades"],
        }

        logger.info("Análise completa executada e arquivos gerados.")
        return resultados

