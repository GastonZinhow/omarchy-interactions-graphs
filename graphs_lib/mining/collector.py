import requests
import csv
import time
from pathlib import Path

GITHUB_API = "https://api.github.com"


# ============================================================
# AUXILIARES
# ============================================================


def auth_headers(token: str):
    return {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}


def safe_get(url: str, headers: dict, params=None):
    """
    GET com retry simples contra rate-limit.
    """
    while True:
        try:
            res = requests.get(url, headers=headers, params=params)
        except Exception as e:
            print("[ERRO DE REDE]", e)
            time.sleep(3)
            continue

        if res.status_code == 200:
            return res.json()

        if res.status_code == 403:
            print("[RATE LIMIT] Aguardando 8 segundos...")
            time.sleep(8)
            continue

        print("[ERRO]", res.status_code, res.text)
        return None


def paginate(url: str, headers: dict, params=None):
        """
        Paginação com logs simples.
        """
        results = []
        page = 1
        print(f"\n[+] Paginação: {url}")

        while True:
            p = params or {}
            p["page"] = page
            p["per_page"] = 100

            print(f"  - Página {page}")
            data = safe_get(url, headers, p)

            if not data or len(data) == 0:
                print("  - Fim da paginação.")
                break

            print(f"    {len(data)} itens.")
            results.extend(data)
            page += 1
            time.sleep(0.4)

        print(f"[TOTAL] {len(results)} itens coletados\n")
        return results

# ============================================================
# COLETA OTIMIZADA
# ============================================================

def collect_all(owner, repo, token):

        headers = auth_headers(token)

        print("=== COLETANDO ISSUES ===")
        issues = paginate(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues", headers, params={"state": "all"}
        )

        # cache: evita milhares de GETs
        issue_author_map = {
            str(issue["number"]): issue["user"]["login"] for issue in issues
        }

        print("=== COLETANDO COMENTÁRIOS DE ISSUES ===")
        issue_comments = paginate(
            f"{GITHUB_API}/repos/{owner}/{repo}/issues/comments", headers
        )

        print("=== COLETANDO PULL REQUESTS ===")
        pulls = paginate(
            f"{GITHUB_API}/repos/{owner}/{repo}/pulls", headers, params={"state": "all"}
        )

        pr_author_map = {str(pr["number"]): pr["user"]["login"] for pr in pulls}

        print("=== COLETANDO COMENTÁRIOS DE PULL REQUESTS ===")
        pr_comments = paginate(f"{GITHUB_API}/repos/{owner}/{repo}/pulls/comments", headers)

        return (
            issues,
            issue_comments,
            pulls,
            pr_comments,
            headers,
            pr_author_map,
            issue_author_map,
        )

# ============================================================
# NORMALIZAÇÃO DAS INTERAÇÕES
# ============================================================

def extract_interactions(owner, repo, token):
        edges = []

        (
            issues,
            issue_comments,
            pulls,
            pr_comments,
            headers,
            pr_author_map,
            issue_author_map,
        ) = collect_all(owner, repo, token)

# ----------------------------------------------------------
# Issues — fechamento
# ----------------------------------------------------------
        print("\n=== PROCESSANDO ISSUES ===")
        for issue in issues:
            author = issue["user"]["login"]

            if issue.get("closed_by"):
                closer = issue["closed_by"]["login"]
                if closer != author:
                    edges.append((closer, author, "fechamento_issue", 3))
                    print(f"[EDGE] {closer} -> {author} (fechamento issue)")

    # ----------------------------------------------------------
    # Comentários de issues (SEM requisições extras)
    # ----------------------------------------------------------
        print("\n=== PROCESSANDO COMENTÁRIOS DE ISSUES ===")
        for c in issue_comments:
            commenter = c["user"]["login"]
            issue_number = c["issue_url"].split("/")[-1]

            if issue_number in issue_author_map:
                target = issue_author_map[issue_number]
                if commenter != target:
                    edges.append((commenter, target, "comentario", 2))
                    print(f"[EDGE] {commenter} -> {target} (comentário issue)")

    # ----------------------------------------------------------
    # Comentários de PR (SEM requisições extras)
    # ----------------------------------------------------------
        print("\n=== PROCESSANDO COMENTÁRIOS DE PR ===")
        for c in pr_comments:
            commenter = c["user"]["login"]
            pr_number = c["pull_request_url"].split("/")[-1]

            if pr_number in pr_author_map:
                target = pr_author_map[pr_number]
                if commenter != target:
                    edges.append((commenter, target, "comentario", 2))
                    print(f"[EDGE] {commenter} -> {target} (comentário PR)")

    # ----------------------------------------------------------
    # Reviews + merges (ZERO chamadas extras)
    # ----------------------------------------------------------
        print("\n=== PROCESSANDO REVIEWS E MERGES ===")
        for pr in pulls:
            pr_number = pr["number"]
            pr_author = pr["user"]["login"]

            # Reviews
            reviews_url = f"{GITHUB_API}/repos/{owner}/{repo}/pulls/{pr_number}/reviews"
            reviews = safe_get(reviews_url, headers) or []

            for r in reviews:
                reviewer = r["user"]["login"]
                if reviewer != pr_author:
                    edges.append((reviewer, pr_author, "review", 4))
                    print(f"[EDGE] {reviewer} -> {pr_author} (review)")

            # Merge (USANDO DADOS DO PR, sem GET extra)
            if pr.get("merged_at"):
                merger = pr.get("merged_by", {}).get("login")
                if merger and merger != pr_author:
                    edges.append((merger, pr_author, "merge", 5))
                    print(f"[EDGE] {merger} -> {pr_author} (merge)")

            time.sleep(0.2)

        print("\n=== FINALIZADO ===")
        print(f"TOTAL DE ARESTAS: {len(edges)}\n")

        return edges

# ============================================================
# CATEGORIZAÇÃO
# ============================================================

def categorize_edges(edges):
    comentarios = []
    fechamentos = []
    reviews_merges = []
    for src, tgt, tipo, peso in edges:
        if tipo == "comentario":
            comentarios.append((src, tgt, tipo, peso))
        elif tipo == "fechamento_issue":
            fechamentos.append((src, tgt, tipo, peso))
        elif tipo in ("review", "merge"):
            reviews_merges.append((src, tgt, tipo, peso))
        return {
            "comentarios": comentarios,
            "fechamentos": fechamentos,
            "reviews_merges": reviews_merges,
            }

# ============================================================
# CSV
# ============================================================

def save_csv(edges, output_path):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "target", "tipo_interacao", "peso"])
            w.writerows(edges)
        print(f"[CSV] Arquivo salvo em: {output_path}")

def save_csv_split(categorias_dict, output_dir):
        """Exporta três CSVs separados com nomes padronizados para o diretório fornecido."""
        base = Path(output_dir)
        base.mkdir(parents=True, exist_ok=True)
        nomes = {
            "comentarios": "interacoes_comentarios.csv",
            "fechamentos": "interacoes_fechamentos.csv",
            "reviews_merges": "interacoes_reviews_merges.csv",
        }
        for cat, edges in categorias_dict.items():
            out_path = base / nomes[cat]
            save_csv(edges, out_path)
        # Também exporta um CSV combinado com todas as interações
        combined = []
        for lst in categorias_dict.values():
            combined.extend(lst)
        combined_path = base / "interacoes_todas.csv"
        save_csv(combined, combined_path)

# ============================================================
# MAIN
# ============================================================

def main():
        import argparse

        parser = argparse.ArgumentParser(description="Coletor de interações do GitHub (gera 3 CSVs categorizados)")
        parser.add_argument("--owner", required=True, help="Organização ou usuário dono do repositório")
        parser.add_argument("--repo", required=True, help="Nome do repositório")
        parser.add_argument("--token", required=True, help="Personal Access Token do GitHub")
        parser.add_argument(
            "--output",
            default="collected-data",
            help="Diretório de saída dos CSVs categorizados (padrão: collected-data)",
        )
        args = parser.parse_args()

        final_dir = Path(args.output)

        edges = extract_interactions(args.owner, args.repo, args.token)
        categorias = categorize_edges(edges)
        print(f"[INFO] Exportando CSVs separados (comentarios, fechamentos, reviews_merges) para '{final_dir}'...")
        save_csv_split(categorias, final_dir)
        print("[OK] Coleta concluída.")

        if __name__ == "__main__":
            main()