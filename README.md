# 📊 Sistema de Análise de Redes de Colaboração GitHub

Sistema desenvolvido para analisar estruturas de colaboração e interação entre participantes de um projeto no GitHub, como parte do trabalho prático da disciplina de Teoria de Grafos e Computabilidade – PUC Minas.

[](https://www.python.org/)
[](https://networkx.org/)
[](LICENSE)

**Repositório Analisado:** [rails/rails](https://github.com/basecamp/omarchy?tab=readme-ov-file) (55.000+ ⭐)

---

### Como executar o Projeto


````bash
# 1. Instalar dependências
pip install -r requirements.txt

# 2. Executar análise completa
python -m graphs_lib analyze --interactions collected-data/interacoes_todas.csv

# 3. Executar análise para cada tipo de interacao
python -m graphs_lib analyze --interactions collected-data/interacoes_comentarios.csv
python -m graphs_lib analyze --interactions collected-data/interacoes_fechamentos.csv
python -m graphs_lib analyze --interactions collected-data/interacoes_reviews_merges.csv

📂 Arquivos Gerados
Após a execução (exemplo para saída por tipo de interação), você encontrará uma pasta `output/` contendo subpastas por conjunto de interações:

output/
├── interacoes_comentarios/
│   ├── centralidades_interacoes.csv
│   ├── centralidades_normal.csv
│   ├── comunidades.csv
│   ├── grafo_interacoes.gexf
│   ├── grafo_normal.gexf
│   └── metricas_estruturais.csv
├── interacoes_fechamentos/
│   ├── centralidades_interacoes.csv
│   ├── centralidades_normal.csv
│   ├── comunidades.csv
│   ├── grafo_interacoes.gexf
│   ├── grafo_normal.gexf
│   └── metricas_estruturais.csv
└── interacoes_reviews_merges/
	├── centralidades_interacoes.csv
	├── centralidades_normal.csv
	├── comunidades.csv
	├── grafo_interacoes.gexf
	├── grafo_normal.gexf
	└── metricas_estruturais.csv

---

### Coleta de Dados (Minerador GitHub)

O coletor em `graphs_lib/mining/collector.py` gera três conjuntos separados (comentários, fechamentos de issues e reviews/merges) em CSV e também um CSV combinado com todas as interações (`interacoes_todas.csv`).

Pré-requisitos:
- **Personal Access Token** do GitHub (escopo público básico para leitura de issues e PRs).
- Acesso de rede para chamadas à API.

#### Executar coleta

```powershell
python -m graphs_lib.mining.collector --owner ORGANIZACAO --repo REPOSITORIO --token SEU_TOKEN --output collected-data
````

Use `--output` para definir a pasta de saída. Serão gerados na pasta escolhida:

- `interacoes_comentarios.csv`
- `interacoes_fechamentos.csv`
- `interacoes_reviews_merges.csv`
- `interacoes_todas.csv` (combina as três categorias)

Categorias:

- `comentarios`: comentários em issues ou pull requests
- `fechamentos`: fechamento de issues por usuário diferente do autor
- `reviews_merges`: reviews e merges de pull requests

#### Executar análise em um dos conjuntos

Escolha um dos CSVs gerados e rode:

```powershell
python -m graphs_lib analyze --interactions collected-data/interacoes_comentarios.csv
```

Repita para os demais CSVs se quiser análises independentes por tipo. Cada execução da análise gera os arquivos de métricas e os grafos `.gexf` correspondentes ao conjunto fornecido.

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
|
collected-data/
│   ├── interacoes_comentarios.csv
│   ├── interacoes_fechamentos.csv
│   ├── interacoes_reviews_merges.csv
│   └── interacoes_todas.csv
│
logs/
│   └── graphs_lib.log
│
├── interacoes.csv
├── .gitignore
└── README.md
```
