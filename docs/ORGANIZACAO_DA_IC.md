# Organização dos materiais da IC — ERAD-SE 2026

Este repositório corresponde ao estudo **Avaliação do Custo de Treinamento de IDS para IoT em Clusters CPU e GPU**, relacionado à iniciação científica PIBIC/CNPq 2025–2026.

## Materiais

| Conteúdo | Local |
| --- | --- |
| Código do experimento | `src/erad_pipeline/`, `run_experiment.py` |
| Submissões PBS/Slurm | `scripts/` |
| Instruções técnicas | `README_ERAD_EXPERIMENT.md` |
| Artigo, fontes e pôster | `publicacoes/erad-se-2026/` (a incluir) |
| Figuras selecionadas e legendas | `figuras/` (a incluir) |
| Tempos, métricas e metadados compactos | `resultados/` (a incluir) |

Não versionar datasets brutos, checkpoints, logs de cluster ou saídas volumosas. Registrar parâmetros, ambiente, identificadores das execuções e diferenças metodológicas entre CPU e GPU.

**Atenção à interpretação:** o caminho GPU acelerado não é metodologicamente idêntico ao CPU; diferenças de codificação e implementação impedem atribuir toda a aceleração exclusivamente ao hardware.

## Trabalhos relacionados

- [SBCUP 2026 — abordagem em duas fases](https://github.com/Math-BUG/iot-fog-ids-two-phase)
- [Generalização e avaliação em maior escala](https://github.com/Math-BUG/ids_generalization_pipeline)

A pesquisa é relacionada, mas os experimentos e respectivos protocolos de avaliação devem ser documentados separadamente.
