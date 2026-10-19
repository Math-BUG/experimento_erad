# Avaliação do Custo de Treinamento de IDS para IoT em Clusters CPU e GPU

Repositório dos experimentos de **custo computacional (CPU e GPU)** relacionados ao trabalho apresentado no **ERAD-SE 2026**, desenvolvido na iniciação científica PIBIC/CNPq da Universidade Federal de Viçosa.

**Autores:** Matheus Antony Souza Pereira, Vitor Barbosa Souza e Daniel Louzada Fernandes.

## Objetivo

Investigar o tempo de execução de um pipeline de IDS para tráfego IoT em ambientes CPU e GPU, incluindo preparação de dados, clustering e treinamento supervisionado. Os experimentos usam dados TON_IoT e scripts próprios de execução em cluster.

**Nota metodológica:** as versões CPU e GPU acelerada apresentam diferenças de implementação e pré-processamento. As medidas de tempo observadas não isolam apenas o efeito do hardware. Consulte a metodologia antes de comparar números.

## Onde encontrar

| Material | Caminho |
| --- | --- |
| Pipeline Python | [`src/erad_pipeline/`](src/erad_pipeline/) |
| Executável | [`run_experiment.py`](run_experiment.py) |
| Scripts PBS/Slurm | [`scripts/`](scripts/) |
| Procedimento de reprodução e parâmetros | [`README_ERAD_EXPERIMENT.md`](README_ERAD_EXPERIMENT.md) |
| Publicação e pôster (organização preparada) | [`publicacoes/erad-se-2026/`](publicacoes/erad-se-2026/) |
| Gráficos selecionados | [`figuras/`](figuras/) |
| Resultados consolidados | [`resultados/`](resultados/) |
| Plano de organização | [`docs/ORGANIZACAO_DA_IC.md`](docs/ORGANIZACAO_DA_IC.md) |

**Os arquivos da publicação e as figuras finais ainda precisam ser adicionados.** Os diretórios de destino foram preparados sem alterar o código e os scripts atuais.

## Parte da mesma iniciação científica

- [SBCUP 2026 — abordagem em duas fases para detecção e classificação de ataques](https://github.com/Math-BUG/iot-fog-ids-two-phase)
- [Generalização e avaliação de IDS em maior escala](https://github.com/Math-BUG/ids_generalization_pipeline)

Os três repositórios têm objetivos experimentais diferentes e suas métricas não devem ser misturadas.
