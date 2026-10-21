"""
Registrador de eventos (padrão Observer em memória).

Implementa um pub/sub leve, sem dependências externas, usado para
rastrear a execução do pipeline (RNF11 - Auditabilidade), sem registrar
dados individuais do dataset (RNF06 - Segurança). Cada evento é gravado
em um arquivo JSON Lines (um objeto JSON por linha), de modo que o
histórico completo da execução possa ser auditado posteriormente.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, DefaultDict, List


class RegistradorDeEventos:
    """Observer em memória: publica eventos e os grava em log estruturado."""

    def __init__(self, caminho_log: str | Path = "results/eventos.jsonl"):
        self.caminho_log = Path(caminho_log)
        self.caminho_log.parent.mkdir(parents=True, exist_ok=True)
        self._assinantes: DefaultDict[str, List[Callable[[dict], None]]] = defaultdict(list)
        # reinicia o arquivo de log a cada execução do pipeline
        self.caminho_log.write_text("", encoding="utf-8")

    def assinar(self, nome_evento: str, callback: Callable[[dict], None]) -> None:
        """Registra um assinante (callback) para um tipo de evento."""
        self._assinantes[nome_evento].append(callback)

    def publicar(self, nome_evento: str, metadados: dict[str, Any] | None = None) -> None:
        """Publica um evento: grava no log e notifica os assinantes."""
        registro = {
            "evento": nome_evento,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "metadados": metadados or {},
        }
        with self.caminho_log.open("a", encoding="utf-8") as f:
            f.write(json.dumps(registro, ensure_ascii=False) + "\n")
        for callback in self._assinantes.get(nome_evento, []):
            callback(registro)
        print(f"[evento] {nome_evento} :: {metadados or {}}")
