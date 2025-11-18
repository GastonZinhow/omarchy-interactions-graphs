import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import community as community_louvain

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class AnaliseRedeGitHub:
    
    def __init__(self, arquivo_grafo=None, arquivo_interacoes=None):

        self.G_normal = None
        self.G_interacoes = None
        
        if arquivo_grafo:
            self.carregar_grafo_normal(arquivo_grafo)
        if arquivo_interacoes:
            self.carregar_grafo_interacoes(arquivo_interacoes)
    
    def carregar_grafo_normal(self, arquivo):
        df = pd.read_csv(arquivo)
        self.G_normal = nx.from_pandas_edgelist(
            df, 
            source='source', 
            target='target', 
            edge_attr='peso',
            create_using=nx.Graph()
        )
        print(f"✓ Grafo normal carregado: {self.G_normal.number_of_nodes()} nós, {self.G_normal.number_of_edges()} arestas")
    
    def carregar_grafo_interacoes(self, arquivo):
        df = pd.read_csv(arquivo)
        self.G_interacoes = nx.from_pandas_edgelist(
            df,
            source='source',
            target='target',
            edge_attr=['tipo_interacao', 'peso'],
            create_using=nx.MultiDiGraph()
        )
        print(f"✓ Grafo de interações carregado: {self.G_interacoes.number_of_nodes()} nós, {self.G_interacoes.number_of_edges()} arestas")
    
    def calcular_centralidades(self, grafo, nome_grafo=""):
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DE CENTRALIDADE - {nome_grafo}")
        print(f"{'='*60}")
        
        # 1. Degree Centrality
        degree_cent = nx.degree_centrality(grafo)
        
        # 2. Betweenness Centrality
        print("Calculando Betweenness Centrality...")
        betweenness = nx.betweenness_centrality(grafo, weight='peso')
        
        # 3. Closeness Centrality
        print("Calculando Closeness Centrality...")
        closeness = nx.closeness_centrality(grafo, distance='peso')
        
        # 4. PageRank
        print("Calculando PageRank...")
        pagerank = nx.pagerank(grafo, weight='peso')
        
        # 5. Eigenvector Centrality
        print("Calculando Eigenvector Centrality...")
        try:
            eigenvector = nx.eigenvector_centrality(grafo, weight='peso', max_iter=1000)
        except:
            eigenvector = {}
            print("⚠️ Eigenvector não convergiu, usando valores vazios")
        
        # Organizar resultados
        resultados = pd.DataFrame({
            'colaborador': list(degree_cent.keys()),
            'degree': list(degree_cent.values()),
            'betweenness': [betweenness.get(n, 0) for n in degree_cent.keys()],
            'closeness': [closeness.get(n, 0) for n in degree_cent.keys()],
            'pagerank': [pagerank.get(n, 0) for n in degree_cent.keys()],
            'eigenvector': [eigenvector.get(n, 0) for n in degree_cent.keys()]
        })
        
        resultados = resultados.sort_values('pagerank', ascending=False)
        
        # Exibir Top 10
        print(f"\n🏆 TOP 10 COLABORADORES MAIS INFLUENTES (PageRank):")
        print(resultados.head(10)[['colaborador', 'pagerank', 'degree', 'betweenness']].to_string(index=False))
        
        return resultados
    
    # ==================== MÉTRICAS DE ESTRUTURA ====================
    
    def analisar_estrutura(self, grafo, nome_grafo=""):
        print(f"\n{'='*60}")
        print(f"MÉTRICAS DE ESTRUTURA E COESÃO - {nome_grafo}")
        print(f"{'='*60}")
        
        # 1. Densidade da rede
        densidade = nx.density(grafo)
        print(f"Densidade da rede: {densidade:.4f}")
        
        # 2. Coeficiente de aglomeração
        clustering_global = nx.average_clustering(grafo, weight='peso')
        print(f"Coeficiente de aglomeração médio: {clustering_global:.4f}")
        
        # 3. Assortatividade
        try:
            assortatividade = nx.degree_assortativity_coefficient(grafo)
            print(f"Assortatividade: {assortatividade:.4f}")
            if assortatividade > 0:
                print("  → Rede assortativa: colaboradores conectados tendem a ter grau similar")
            else:
                print("  → Rede dissortativa: colaboradores com muitas conexões se ligam a poucos conectados")
        except:
            print("⚠️ Não foi possível calcular assortatividade")
        
        # 4. Componentes conexos
        if isinstance(grafo, nx.Graph):
            n_componentes = nx.number_connected_components(grafo)
            maior_componente = max(nx.connected_components(grafo), key=len)
            print(f"\nNúmero de componentes conexos: {n_componentes}")
            print(f"Tamanho do maior componente: {len(maior_componente)} ({100*len(maior_componente)/grafo.number_of_nodes():.1f}%)")
        
        # 5. Distribuição de grau
        graus = [grafo.degree(n) for n in grafo.nodes()]
        print(f"\nDistribuição de grau:")
        print(f"  - Grau médio: {np.mean(graus):.2f}")
        print(f"  - Grau mediano: {np.median(graus):.2f}")
        print(f"  - Grau máximo: {max(graus)}")
        print(f"  - Grau mínimo: {min(graus)}")
        
        return {
            'densidade': densidade,
            'clustering': clustering_global,
            'assortatividade': assortatividade if 'assortatividade' in locals() else None,
            'graus': graus
        }
    
    # ==================== DETECÇÃO DE COMUNIDADES ====================
    
    def detectar_comunidades(self, grafo, nome_grafo=""):
        """Detecta comunidades usando algoritmo de Louvain"""
        print(f"\n{'='*60}")
        print(f"DETECÇÃO DE COMUNIDADES - {nome_grafo}")
        print(f"{'='*60}")
        
        # Algoritmo de Louvain
        particoes = community_louvain.best_partition(grafo, weight='peso')
        
        # Calcular modularidade
        modularidade = community_louvain.modularity(particoes, grafo, weight='peso')
        print(f"Modularidade: {modularidade:.4f}")
        
        # Estatísticas das comunidades
        comunidades_counter = Counter(particoes.values())
        n_comunidades = len(comunidades_counter)
        print(f"Número de comunidades detectadas: {n_comunidades}")
        
        print(f"\nTamanho das 10 maiores comunidades:")
        for i, (com_id, tamanho) in enumerate(comunidades_counter.most_common(10), 1):
            print(f"  {i}. Comunidade {com_id}: {tamanho} membros")
        
        # Identificar bridging ties (nós que conectam comunidades)
        bridging_scores = {}
        for node in grafo.nodes():
            vizinhos = list(grafo.neighbors(node))
            if len(vizinhos) > 0:
                comunidades_vizinhas = [particoes[v] for v in vizinhos]
                diversidade = len(set(comunidades_vizinhas)) / len(comunidades_vizinhas)
                bridging_scores[node] = diversidade
        
        top_bridges = sorted(bridging_scores.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n🌉 TOP 10 BRIDGING TIES (conectores de comunidades):")
        for node, score in top_bridges:
            print(f"  - {node}: {score:.3f}")
        
        return {
            'particoes': particoes,
            'modularidade': modularidade,
            'n_comunidades': n_comunidades,
            'bridging_ties': bridging_scores
        }
    
    # ==================== VISUALIZAÇÕES ====================
    
    def visualizar_metricas(self, centralidades, estrutura, titulo=""):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Análise de Métricas - {titulo}', fontsize=16, fontweight='bold')
        
        # 1. Top 20 PageRank
        top20 = centralidades.nlargest(20, 'pagerank')
        axes[0, 0].barh(range(len(top20)), top20['pagerank'])
        axes[0, 0].set_yticks(range(len(top20)))
        axes[0, 0].set_yticklabels(top20['colaborador'], fontsize=8)
        axes[0, 0].set_xlabel('PageRank')
        axes[0, 0].set_title('Top 20 - PageRank')
        axes[0, 0].invert_yaxis()
        
        # 2. Top 20 Betweenness
        top20_bet = centralidades.nlargest(20, 'betweenness')
        axes[0, 1].barh(range(len(top20_bet)), top20_bet['betweenness'])
        axes[0, 1].set_yticks(range(len(top20_bet)))
        axes[0, 1].set_yticklabels(top20_bet['colaborador'], fontsize=8)
        axes[0, 1].set_xlabel('Betweenness')
        axes[0, 1].set_title('Top 20 - Betweenness (Pontes)')
        axes[0, 1].invert_yaxis()
        
        # 3. Distribuição de Grau
        axes[0, 2].hist(estrutura['graus'], bins=50, edgecolor='black')
        axes[0, 2].set_xlabel('Grau')
        axes[0, 2].set_ylabel('Frequência')
        axes[0, 2].set_title('Distribuição de Grau')
        axes[0, 2].set_yscale('log')
        
        # 4. Scatter: Degree vs PageRank
        axes[1, 0].scatter(centralidades['degree'], centralidades['pagerank'], alpha=0.5)
        axes[1, 0].set_xlabel('Degree Centrality')
        axes[1, 0].set_ylabel('PageRank')
        axes[1, 0].set_title('Degree vs PageRank')
        
        # 5. Scatter: Betweenness vs Closeness
        axes[1, 1].scatter(centralidades['betweenness'], centralidades['closeness'], alpha=0.5)
        axes[1, 1].set_xlabel('Betweenness')
        axes[1, 1].set_ylabel('Closeness')
        axes[1, 1].set_title('Betweenness vs Closeness')
        
        # 6. Box plot das centralidades
        dados_box = centralidades[['degree', 'betweenness', 'closeness', 'pagerank']].values
        axes[1, 2].boxplot(dados_box, labels=['Degree', 'Betweenness', 'Closeness', 'PageRank'])
        axes[1, 2].set_ylabel('Valor normalizado')
        axes[1, 2].set_title('Distribuição das Centralidades')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig
    
    def exportar_gephi(self, grafo, arquivo_saida, comunidades=None):
        """Exporta o grafo para Gephi (.gexf)"""
        if comunidades:
            for node, com_id in comunidades['particoes'].items():
                grafo.nodes[node]['comunidade'] = com_id
        
        nx.write_gexf(grafo, arquivo_saida)
        print(f"✓ Grafo exportado para Gephi: {arquivo_saida}")
    
    # ==================== PIPELINE COMPLETO ====================
    
    def executar_analise_completa(self):
        """Executa análise completa em ambos os grafos"""
        resultados = {}
        
        if self.G_normal:
            print("\n" + "="*70)
            print("ANÁLISE DO GRAFO DE COLABORAÇÃO GERAL")
            print("="*70)
            
            cent_normal = self.calcular_centralidades(self.G_normal, "Grafo Normal")
            estr_normal = self.analisar_estrutura(self.G_normal, "Grafo Normal")
            com_normal = self.detectar_comunidades(self.G_normal, "Grafo Normal")
            
            fig1 = self.visualizar_metricas(cent_normal, estr_normal, "Grafo de Colaboração")
            plt.savefig('analise_grafo_normal.png', dpi=300, bbox_inches='tight')
            print("✓ Visualizações salvas: analise_grafo_normal.png")
            
            self.exportar_gephi(self.G_normal, 'grafo_normal.gexf', com_normal)
            
            resultados['normal'] = {
                'centralidades': cent_normal,
                'estrutura': estr_normal,
                'comunidades': com_normal
            }
        
        if self.G_interacoes:
            print("\n" + "="*70)
            print("ANÁLISE DO GRAFO DE INTERAÇÕES")
            print("="*70)
            
            # Converter para grafo simples para análise
            G_inter_simples = nx.Graph(self.G_interacoes)
            
            cent_inter = self.calcular_centralidades(G_inter_simples, "Grafo de Interações")
            estr_inter = self.analisar_estrutura(G_inter_simples, "Grafo de Interações")
            com_inter = self.detectar_comunidades(G_inter_simples, "Grafo de Interações")
            
            fig2 = self.visualizar_metricas(cent_inter, estr_inter, "Grafo de Interações")
            plt.savefig('analise_grafo_interacoes.png', dpi=300, bbox_inches='tight')
            print("✓ Visualizações salvas: analise_grafo_interacoes.png")
            
            self.exportar_gephi(G_inter_simples, 'grafo_interacoes.gexf', com_inter)
            
            resultados['interacoes'] = {
                'centralidades': cent_inter,
                'estrutura': estr_inter,
                'comunidades': com_inter
            }
        
        return resultados

if __name__ == "__main__":
    analise = AnaliseRedeGitHub(
        arquivo_grafo='grafo_exportado.csv',
        arquivo_interacoes='interacoes.csv'
    )
    
    # Executar análise completa
    resultados = analise.executar_analise_completa()
    
    if 'normal' in resultados:
        resultados['normal']['centralidades'].to_csv('centralidades_normal.csv', index=False)
        print("✓ Métricas salvas: centralidades_normal.csv")
    
    if 'interacoes' in resultados:
        resultados['interacoes']['centralidades'].to_csv('centralidades_interacoes.csv', index=False)
        print("✓ Métricas salvas: centralidades_interacoes.csv")
    
    print("\n" + "="*70)
    print("ANÁLISE CONCLUÍDA!")
    print("="*70)
    print("Arquivos gerados:")
    print("  - analise_grafo_normal.png")
    print("  - analise_grafo_interacoes.png")
    print("  - grafo_normal.gexf (importar no Gephi)")
    print("  - grafo_interacoes.gexf (importar no Gephi)")
    print("  - centralidades_normal.csv")
    print("  - centralidades_interacoes.csv")