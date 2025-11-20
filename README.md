# 📊 Sistema de Análise de Redes de Colaboração GitHub

Sistema desenvolvido para analisar estruturas de colaboração e interação entre participantes de um projeto no GitHub, como parte do trabalho prático da disciplina de Teoria de Grafos e Computabilidade – PUC Minas.

[](https://www.python.org/)
[](https://networkx.org/)
[](LICENSE)

**Repositório Analisado:** [rails/rails](https://github.com/basecamp/omarchy?tab=readme-ov-file) (55.000+ ⭐)

---

### Como executar o Projeto

```bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar análise completa
python -m graphs_lib analyze --interactions interacoes.csv

📂 Arquivos Gerados
Após a execução, você encontrará:

output/analise_grafo_normal.png	Painel com 6 gráficos de métricas
output/analise_grafo_interacoes.png	Análise do grafo direcionado
output/grafo_normal.gexf	Arquivo para importar no Gephi
output/grafo_interacoes.gexf	Grafo direcionado para Gephi
output/centralidades_normal.csv	Métricas de todos os colaboradores
output/centralidades_interacoes.csv	Análise detalhada de interações

---

### Como executar os Testes

```bash
# 1. Executar todos os testes
 python -m unittest discover -s graphs_lib/tests


---

📁 Estrutura do Projeto
graphs_lib/
├── graphs/                          
│   ├── __init__.py                  
│   ├── abstract_graph.py                      
│   ├── adjacency_list_graph.py           
│   ├── adjacency_matrix_graph.py          
│   └── exceptions.py                
│
├── mining/                       
│   ├── __init__.py
│   └── collector.py         
│
├── analysis/                       
│   ├── __init__.py
│   └── network_analysis.py      
│
├── utils/                        
│   ├── __init__.py
│   ├── logger.py                  
│   └── json_utils.py                      
│
├── tests/                          
│   ├── __init__.py
│   ├── test_adjacency_list.py       
│   └── test_adjacency_matrix.py      
│
│   requirements.txt
│   main.py
│   .env.example
│   __main__.py
│   __init__.py   
│  
output/                        
│   ├── analise_grafo_normal.png
│   ├── analise_grafo_interacoes.png
│   ├── grafo_normal.gexf
│   ├── grafo_interacoes.gexf
│   ├── centralidades_normal.csv
│   └── centralidades_interacoes.csv
│
logs/                           
│   └── graphs_lib.log
│
├── interacoes.csv                                                    
├── .gitignore                      
└── README.md                        
