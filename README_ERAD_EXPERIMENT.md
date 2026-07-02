# Experimento ERAD-SE CPU/GPU

Este diretorio e separado dos projetos-fonte. Ele usa como referencia:

- CPU/original/dados: `C:\Users\AlphaXD\Desktop\UFV\IC\Parte-2`
- GPU/generalizado: `C:\Users\AlphaXD\Desktop\UFV\IC\Estendido\ids_generalization_pipeline`

As pastas acima sao fontes de referencia. O codigo executavel do experimento
fica aqui.

## Datasets

O pipeline aceita:

- arquivo Parquet unico, por exemplo `data/train_test_network.parquet`;
- diretorio Parquet particionado, por exemplo `data/TON_IoT_parquet/`;
- CSV, apenas como fallback de compatibilidade.

Organizacao recomendada no cluster:

```text
experimento_erad/
|-- data/
|   |-- train_test_network.parquet
|   `-- TON_IoT_parquet/
|       |-- part-000.parquet
|       `-- ...
|-- run_experiment.py
|-- scripts/
`-- src/
```

`data/` esta no `.gitignore` e nao deve ser versionado.

## Metodologia Preservada

O fluxo segue a versao CPU em `Parte-2/pipeline_meshi_local.py`:

1. leitura dos dados;
2. split treino/validacao/teste com `StratifiedGroupKFold`;
3. pre-processamento;
4. clusterizacao;
5. treinamento supervisionado;
6. predicao;
7. avaliacao;
8. salvamento dos tempos e metricas.

Pontos fixos para comparacao justa:

- Targets supervisionados: `cluster_kmeans30`, `type`, `label`
- Split: grupos por `src_ip`, `dst_ip`, `service`, `proto`
- Seed principal: `42`
- Seed validacao: `123`
- Random Forest: `random_state=10000`, `n_estimators=150`
- KMeans: `k=30`
- SVD: `n_components=50`
- One-hot: `handle_unknown="ignore"`, `min_frequency=10`

## Backends

CPU:

- leitura com pandas;
- pre-processamento com pandas/scikit-learn;
- clusterizacao com `sklearn MiniBatchKMeans`;
- supervisionado com `sklearn RandomForestClassifier`.

GPU:

- leitura tenta cuDF e cai para pandas se necessario;
- clusterizacao tenta `cuML KMeans`;
- `--gpu-mode accelerated` usa CuPy/cuML no pre-processamento numerico,
  codifica categoricas por ordinal limitado por frequencia, usa reducao
  dimensional cuML e treina `cuML RandomForestClassifier`;
- `--gpu-mode strict` mantem pre-processamento e Random Forest em sklearn para
  maior paridade metodologica com CPU;
- split e avaliacao continuam em CPU para manter os mesmos indices e metricas.

Essas decisoes sao registradas em `backend_report.csv`.

O modo acelerado e o mais rapido, mas nao e representacao-identico ao CPU:
categoricas deixam de ser one-hot e passam a ser codigos ordinais limitados por
frequencia, e o RandomForest cuML nao usa
`class_weight="balanced_subsample"`.

## Execucao Local

Instale dependencias CPU:

```bash
cd C:\Users\AlphaXD\Desktop\UFV\IC\ERAD\experimento_erad
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

No cluster GPU, instale as dependencias RAPIDS/CUDA 12 em uma `.venv`
separada:

```bash
cd /home/es112688/experimento_erad
module --force purge
module load GCCcore/12.2.0
module load CUDA/12.6.0

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-gpu-cu12.txt
python -c "import cudf, cuml, cupy; print('GPU deps ok')"
python -m pip freeze > requirements-lock-gpu.txt
```

O arquivo `requirements-lock-gpu.txt` registra as versoes exatas instaladas
na maquina e deve ser guardado junto dos resultados para reprodutibilidade.

Se o pip listar outra versao RAPIDS disponivel no cluster, ajuste
`requirements-gpu-cu12.txt` para a mesma versao em `cudf-cu12` e `cuml-cu12`.
As duas devem ficar alinhadas.

Dataset menor:

```bash
python run_experiment.py --backend cpu --data-path data/train_test_network.parquet --n-samples 1000 --run-id smoke_cpu --output-dir outputs
python run_experiment.py --backend gpu --gpu-mode accelerated --data-path data/train_test_network.parquet --n-samples 1000 --run-id smoke_gpu --output-dir outputs
```

Dataset completo particionado:

```bash
python run_experiment.py --backend cpu --data-path data/TON_IoT_parquet --run-id cpu_full --output-dir outputs
python run_experiment.py --backend gpu --gpu-mode accelerated --data-path data/TON_IoT_parquet --run-id gpu_full --output-dir outputs
```

Use `--n-samples` apenas para testes ou curvas de escala. Para o dataset
inteiro, omita `--n-samples`.

## Cluster CPU PBS

Dataset menor:

```bash
DATA_PATH="$PWD/data/train_test_network.parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="cpu_train_test_network_r1" \
qsub scripts/run_cpu.pbs
```

Dataset completo particionado:

```bash
DATA_PATH="$PWD/data/TON_IoT_parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="cpu_ton_iot_full_r1" \
qsub scripts/run_cpu.pbs
```

Para teste com amostra:

```bash
DATA_PATH="$PWD/data/train_test_network.parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="cpu_100k_r1" \
N_SAMPLES=100000 \
qsub scripts/run_cpu.pbs
```

## Cluster GPU Slurm

Dataset menor:

```bash
DATA_PATH="$PWD/data/train_test_network.parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="gpu_train_test_network_r1" \
GPU_MODE="accelerated" \
sbatch scripts/run_gpu.sbatch
```

Dataset completo particionado:

```bash
DATA_PATH="$PWD/data/TON_IoT_parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="gpu_ton_iot_full_r1" \
GPU_MODE="accelerated" \
sbatch scripts/run_gpu.sbatch
```

Para teste com amostra:

```bash
DATA_PATH="$PWD/data/train_test_network.parquet" \
OUTPUT_DIR="$HOME/dados/erad_experiment_results" \
RUN_ID="gpu_100k_r1" \
N_SAMPLES=100000 \
GPU_MODE="accelerated" \
sbatch scripts/run_gpu.sbatch
```

Para reduzir uso de memoria no modo acelerado, diminua o limite de categorias
mantidas por coluna:

```bash
GPU_MAX_CATEGORIES_PER_COL=32 sbatch scripts/run_gpu.sbatch
```

Para voltar ao caminho conservador com sklearn no pre-processamento/RF:

```bash
GPU_MODE="strict" sbatch scripts/run_gpu.sbatch
```

## Arquivos Gerados

Os resultados ficam em:

```text
<output-dir>/<backend>/<run-id>/
```

Arquivos principais:

- `timings.csv`: tempos por etapa;
- `metrics_quality.csv`: metricas de clusterizacao e classificacao;
- `backend_report.csv`: backend real usado em cada etapa e observacoes;
- `cluster_assignments.csv`: cluster de cada linha;
- `splits/train_indices.csv`, `valid_indices.csv`, `test_indices.csv`;
- `metadata.json`: parametros, fontes e metadados do experimento.

Colunas principais de `timings.csv`:

- `backend`
- `run_id`
- `n_samples`
- `etapa`
- `tempo_segundos`
- `timestamp`
- `nome_do_dataset`
- `implementation_backend`
- `observacoes`

Colunas principais de `metrics_quality.csv`:

- `backend`
- `run_id`
- `n_samples`
- `target`
- `split`
- `accuracy`
- `balanced_accuracy`
- `macro_f1`
- `weighted_f1`
- `mcc`
- `ari`
- `ami`
- `v_measure`
