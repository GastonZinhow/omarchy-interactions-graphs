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
    caminho = Path(caminho)

    if not caminho.exists():
        raise FileNotFoundError(f"❌ Arquivo CSV não encontrado: {caminho}")

    logger.info(f"📄 Carregando CSV de interações: {caminho}")

    interacoes = []
    try:
        with caminho.open("r", encoding="utf-8") as f:
            leitor = csv.reader(f)
            next(leitor)  # pular cabeçalho
            for linha in leitor:
                if len(linha) >= 2:
                    interacoes.append((linha[0], linha[1]))

        logger.info(f"📌 {len(interacoes)} interações carregadas.")
        return interacoes

    except Exception as e:
        logger.error(f"❌ Erro ao carregar CSV: {e}")
        raise

def comando_analyze(args):

    print("\n⏳ Inicializando analisador...\n")

    # Criar pasta output com segurança
    output_dir = garantir_pasta_output()

    # Carregar interações
    interacoes = carregar_interacoes_csv(args.interactions)

    # Instanciar analisador
    analyzer = NetworkAnalyzer(interactions=interacoes, output_dir=output_dir)

    print("\n⏳ Executando análise completa...\n")

    try:
        resultados = analyzer.executar_analise_completa()

        print("✅ Análise concluída com sucesso!\n")

        print("📥 Resultados:")
        for chave, valor in resultados.items():
            print(f"- {chave}: {valor}")

    except Exception as e:
        logger.error(f"❌ Erro na análise: {e}", exc_info=True)
        print(f"\n❌ Erro: {e}")


def main_cli():
    parser = argparse.ArgumentParser(
        description="Ferramenta de análise de repositórios no github"
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
