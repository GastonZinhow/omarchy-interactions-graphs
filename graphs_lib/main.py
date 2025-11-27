import argparse
import logging
import csv
from pathlib import Path
import os

from .analysis.network_analysis import NetworkAnalyzer

logging.basicConfig(
  level=logging.INFO,
  format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

def garantir_pasta_output():
  output_dir = Path("output")

  if not output_dir.exists():
      try:
          output_dir.mkdir(parents=True, exist_ok=True)
          logger.info(f"📁 Pasta 'output/' criada com sucesso.")
      except PermissionError:
          logger.error("❌ Sem permissão para criar a pasta 'output/'.")
          raise
      except Exception as e:
          logger.error(f"❌ Erro ao criar pasta 'output/': {e}")
          raise
  else:
      logger.info("📁 Pasta 'output/' já existe.")

  # Verificar permissão de escrita
  if not os.access(output_dir, os.W_OK):
      raise PermissionError("❌ Sem permissão de escrita na pasta 'output/'.")

  return output_dir

def carregar_interacoes_csv(caminho):
  """
  Carrega CSV de interações no formato:
  source,target,tipo_interacao,peso
  """
  caminho = Path(caminho)

  if not caminho.exists():
      raise FileNotFoundError(f"❌ Arquivo CSV não encontrado: {caminho}")

  logger.info(f"📄 Carregando CSV de interações: {caminho}")

  interacoes = []
  try:
      with caminho.open("r", encoding="utf-8") as f:
          leitor = csv.reader(f)
          header = next(leitor)  # pular cabeçalho
          
          for linha in leitor:
              if len(linha) >= 4:
                  source = linha[0]
                  target = linha[1]
                  tipo = linha[2]
                  peso = int(linha[3])
                  interacoes.append((source, target, tipo, peso))

      logger.info(f"📌 {len(interacoes)} interações carregadas.")
      return interacoes

  except Exception as e:
      logger.error(f"❌ Erro ao carregar CSV: {e}")
      raise

def detectar_tipo_grafo(caminho):
  """Detecta tipo do grafo pelo nome do arquivo."""
  nome = Path(caminho).stem
  
  if "comentarios" in nome:
      return "comentarios"
  elif "fechamentos" in nome:
      return "fechamentos"
  elif "reviews_merges" in nome:
      return "reviews_merges"
  elif "todas" in nome or "integrado" in nome:
      return "grafo_integrado"
  else:
      return "desconhecido"

def comando_analyze(args):
  print("\n⏳ Inicializando analisador...\n")

  # Criar pasta output com segurança
  base_output = garantir_pasta_output()

  # Detectar tipo do grafo
  tipo_grafo = detectar_tipo_grafo(args.interactions)
  
  # Criar subpasta específica para o tipo de análise
  output_dir = base_output / tipo_grafo

  try:
      output_dir.mkdir(parents=True, exist_ok=True)
      logger.info(f"📁 Pasta criada: {output_dir}")
  except Exception as e:
      logger.error(f"❌ Erro ao criar subpasta '{output_dir}': {e}")
      raise

  # Carregar interações
  interacoes = carregar_interacoes_csv(args.interactions)

  # Instanciar analisador
  analyzer = NetworkAnalyzer(
      interactions=interacoes, 
      output_dir=output_dir,
      graph_type=tipo_grafo
  )

  print(f"\n⏳ Executando análise completa do grafo '{tipo_grafo.upper()}'...\n")

  try:
      resultados = analyzer.executar_analise_completa()

      print(f"\n✅ Análise do grafo '{tipo_grafo.upper()}' concluída com sucesso!\n")

      print("📥 Resultados:")
      for chave, valor in resultados.items():
          if isinstance(valor, float):
              print(f"- {chave}: {valor:.4f}")
          else:
              print(f"- {chave}: {valor}")

  except Exception as e:
      logger.error(f"❌ Erro na análise: {e}", exc_info=True)
      print(f"\n❌ Erro: {e}")

def main_cli():
  parser = argparse.ArgumentParser(
      description="Ferramenta de análise de repositórios no GitHub"
  )

  subparsers = parser.add_subparsers(dest="command")

  analyze_parser = subparsers.add_parser("analyze", help="Executa a análise completa")
  analyze_parser.add_argument(
      "--interactions", required=True, help="CSV com interações"
  )

  analyze_parser.set_defaults(func=comando_analyze)

  args = parser.parse_args()

  if hasattr(args, "func"):
      args.func(args)
  else:
      parser.print_help()

if __name__ == "__main__":
  main_cli()