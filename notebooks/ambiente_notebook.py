from __future__ import annotations

import sys
from pathlib import Path


def localizar_raiz_projeto(inicio: Path | None = None) -> Path:
    caminho_atual = (inicio or Path.cwd()).resolve()
    for candidato in [caminho_atual, *caminho_atual.parents]:
        if (candidato / "app.py").exists() and (candidato / "src").exists():
            return candidato
    raise FileNotFoundError("Não foi possível localizar a raiz do projeto a partir do diretório atual.")


def adicionar_raiz_no_syspath(inicio: Path | None = None) -> Path:
    diretorio_raiz = localizar_raiz_projeto(inicio)
    if str(diretorio_raiz) not in sys.path:
        sys.path.insert(0, str(diretorio_raiz))
    return diretorio_raiz


def resolver_caminho_projeto(caminho_relativo: str | Path, inicio: Path | None = None) -> Path:
    return localizar_raiz_projeto(inicio) / Path(caminho_relativo)


def caminho_relativo_projeto(caminho: Path | None, inicio: Path | None = None) -> str:
    if caminho is None:
        return "não localizado"
    diretorio_raiz = localizar_raiz_projeto(inicio)
    caminho_resolvido = caminho.resolve()
    try:
        return caminho_resolvido.relative_to(diretorio_raiz).as_posix()
    except ValueError:
        return str(caminho_resolvido)
