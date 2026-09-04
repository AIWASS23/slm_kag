# SLM-KAG

Implementação de referência local-first da arquitetura proposta em
[`SLM_KAG_SOTA_Research.md`](SLM_KAG_SOTA_Research.md). O sistema combina índice textual,
grafo de propriedades, Personalized PageRank, poda de evidências e um SLM opcional. Toda resposta
expõe as fontes recuperadas e só é marcada como `grounded` quando suas citações são verificáveis.

## O que está implementado

- indexação transacional de documentos e chunks;
- extração de triplas por regras ou por Qwen/Llama/Gemma via Ollama, Hugging Face ou ModelScope;
- índice mútuo `entidade ↔ chunk` para rastrear cada aresta ao texto original;
- busca híbrida: FTS5 + expansão multi-hop + Personalized PageRank;
- orçamento e ranking estrito de evidências para proteger SLMs contra distração;
- forma lógica auditável (`lookup`, `graph_expand`, `count`, `compare`);
- geração determinística offline ou geração local por três backends intercambiáveis;
- verificador de citações e fallback seguro;
- GUI Streamlit, CLI e API FastAPI, com núcleo sem dependências externas.

A descrição de componentes, fluxos e decisões está em
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).
O teste ponta a ponta com os PDFs locais está registrado em
[`docs/REAL_DATA_TEST_REPORT.md`](docs/REAL_DATA_TEST_REPORT.md).

## Início rápido

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[documents]'

slm-kag init
slm-kag ingest examples/company.md
slm-kag query "Onde Ana trabalha?"
```

Também é possível executar sem instalar o pacote:

```bash
PYTHONPATH=src python3 -m slm_kag --db /tmp/slm-kag-demo.db ingest examples/company.md
PYTHONPATH=src python3 -m slm_kag --db /tmp/slm-kag-demo.db query "Onde Ana trabalha?"
```

O Docling normaliza o arquivo antes do chunking. O extrator de fatos por regras reconhece fatos simples e o formato explícito
`sujeito | predicado | objeto`, útil para importações controladas e testes.

## Comandos de execução

Execute os comandos a partir da raiz do projeto:

```bash
cd "/home/marcelo/Área de trabalho/slm_kag"
```

### Ambiente local e `main.py`

Preparação inicial da GUI com Docling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[gui]'
```

Iniciar a interface principal:

```bash
source .venv/bin/activate
streamlit run main.py
```

Comando equivalente depois da instalação:

```bash
slm-kag-gui
```

A interface fica em `http://localhost:8501`.

### API local

Instalar e iniciar a API:

```bash
source .venv/bin/activate
pip install -e '.[api,documents]'
slm-kag serve --host 0.0.0.0 --port 8000
```

Verificar a saúde da API:

```bash
curl --fail http://localhost:8000/health
```

A documentação OpenAPI fica em `http://localhost:8000/docs`.

### Docker Compose

Iniciar somente a GUI:

```bash
docker compose up --build gui
```

Iniciar somente a API:

```bash
docker compose up --build api
```

Iniciar GUI e API em segundo plano:

```bash
docker compose up --build --detach gui api
docker compose ps
docker compose logs --follow gui api
```

Executar com Ollama:

```bash
docker compose --profile slm up --detach ollama
docker compose exec ollama ollama pull qwen2.5:3b
SLM_KAG_EXTRACTOR=ollama SLM_KAG_GENERATOR=ollama \
  docker compose --profile slm up --build gui api
```

Construir a GUI com Hugging Face e ModelScope disponíveis:

```bash
SLM_KAG_GUI_EXTRAS=gui,huggingface,modelscope docker compose build gui
docker compose up gui
```

Encerrar os containers sem apagar os volumes:

```bash
docker compose down
```

### Kubernetes

Primeiro, altere `newName` no `k8s/overlays/<backend>/kustomization.yaml` para a imagem publicada
no seu GHCR. Em seguida, escolha um dos overlays:

```bash
# Regras + resposta determinística
kubectl apply -k k8s/overlays/default

# Hugging Face
kubectl apply -k k8s/overlays/huggingface

# ModelScope
kubectl apply -k k8s/overlays/modelscope

# Ollama com download inicial do Qwen
kubectl apply -k k8s/overlays/ollama
```

Verificar o deploy e os volumes:

```bash
kubectl -n slm-kag rollout status deployment/slm-kag
kubectl -n slm-kag get pods,services,pvc
kubectl -n slm-kag logs deployment/slm-kag -c gui
kubectl -n slm-kag logs deployment/slm-kag -c api
```

Abrir a GUI e a API localmente, usando dois terminais:

```bash
kubectl -n slm-kag port-forward service/slm-kag-gui 8501:80
kubectl -n slm-kag port-forward service/slm-kag-api 8000:8000
```

Remover uma variante sem excluir manualmente os PVCs:

```bash
kubectl delete -k k8s/overlays/default
```

Mais opções e os cuidados relacionados ao SQLite estão em [`k8s/README.md`](k8s/README.md).

## Interface Streamlit

Instale a interface com o Docling e inicie o ponto de entrada `main.py`:

```bash
pip install -e '.[gui]'
streamlit run main.py
```

Acesse `http://localhost:8501`. A interface permite:

- selecionar qualquer quantidade de arquivos ou uma pasta inteira;
- acumular quantos lotes forem necessários no mesmo índice SQLite;
- adicionar metadados JSON a cada lote;
- indexar diretamente todos os documentos compatíveis encontrados em `data/`;
- escolher extração e geração por regras, Ollama, Hugging Face ou ModelScope;
- perguntar somente depois que houver ao menos um documento persistido;
- inspecionar evidências, metadados, plano de consulta e confiança da resposta.

Para habilitar explicitamente os dois hubs de modelos:

```bash
pip install -e '.[gui,huggingface,modelscope]'
streamlit run main.py
```

O banco padrão é `data/slm_kag.db`. Assim, fechar o navegador ou adicionar um novo lote não apaga
os documentos já indexados. Se um arquivo falhar, a interface registra o erro e continua os demais
itens do lote.

Com Docker:

```bash
docker compose up --build gui
```

Para incluir o ModelScope na imagem da GUI, use
`SLM_KAG_GUI_EXTRAS=gui,modelscope docker compose build gui`.

## Kubernetes e CI

Há manifests Kustomize para execução offline, Hugging Face, ModelScope e Ollama em
[`k8s/`](k8s/README.md). Depois de apontar `newName` para a imagem do seu repositório:

```bash
kubectl apply -k k8s/overlays/default
```

GUI e API ficam no mesmo Pod, com volumes persistentes para SQLite e cache de modelos, probes de
saúde e execução sem root. O limite de uma réplica é intencional enquanto o armazenamento for
SQLite.

O workflow [`.github/workflows/ci.yml`](.github/workflows/ci.yml) executa lint, compilação, testes,
renderização/validação dos quatro overlays e build da imagem. Pull requests não publicam artefatos;
pushes em `main`/`master`, tags `v*` e execuções manuais publicam a imagem completa no GHCR com
tags de branch, versão e SHA, SBOM e atestado de proveniência.

## Ingestão documental com Docling

O loader padrão é o Docling e aceita formatos como PDF, DOCX, PPTX, HTML, Markdown e imagens. A
conversão produz Markdown estrutural, preserva separadores de página e registra:

- nome, extensão, MIME type e tamanho do arquivo;
- SHA-256, data de modificação e mudança de metadados;
- `origin`, versão do schema e nome detectados pelo Docling;
- quantidade/números das páginas, status da conversão e erros parciais;
- metadados adicionais enviados pelo usuário ou pela API.

Essas informações são armazenadas em `documents.metadata_json` e retornam no campo `metadata` de
cada evidência e do relatório de indexação. Para um arquivo de texto UTF-8 em ambiente mínimo, o fallback deve ser solicitado
explicitamente com `--loader text` ou `SLM_KAG_DOCUMENT_LOADER=text`; não há fallback silencioso.

## Usar um SLM local

Os backends de extração e geração podem ser escolhidos separadamente. Por exemplo, é possível
extrair com ModelScope e responder com Hugging Face:

```bash
export SLM_KAG_EXTRACTOR=modelscope
export SLM_KAG_GENERATOR=huggingface
```

### Ollama

Com o Ollama em execução e o modelo disponível:

```bash
export SLM_KAG_EXTRACTOR=ollama
export SLM_KAG_GENERATOR=ollama
export SLM_KAG_OLLAMA_MODEL=qwen2.5:3b

slm-kag ingest meu_documento.md
slm-kag query "Minha pergunta multi-hop"
```

A extração usa JSON Schema e temperatura zero. Na resposta, o prompt exige citações `[E1]`,
`[E2]`; se o modelo omiti-las ou inventar identificadores, o resultado é substituído por uma
síntese determinística das evidências.

### Hugging Face

```bash
pip install -e '.[huggingface]'

export SLM_KAG_EXTRACTOR=huggingface
export SLM_KAG_GENERATOR=huggingface
export SLM_KAG_HF_MODEL=Qwen/Qwen2.5-3B-Instruct

slm-kag ingest meu_documento.md
slm-kag query "Minha pergunta"
```

O adaptador usa `AutoTokenizer`, o `chat_template` publicado com o modelo e
`AutoModelForCausalLM`. Pesos são carregados somente na primeira inferência. `device=auto` delega
a distribuição ao Transformers/Accelerate; também é possível definir `cpu`, `cuda` ou `mps`.

### ModelScope

```bash
pip install -e '.[modelscope]'

export SLM_KAG_EXTRACTOR=modelscope
export SLM_KAG_GENERATOR=modelscope
export SLM_KAG_MODELSCOPE_MODEL=Qwen/Qwen2.5-3B-Instruct

slm-kag ingest meu_documento.md
slm-kag query "Minha pergunta"
```

Nesse backend, `snapshot_download` materializa o modelo no cache local e o mesmo runtime
Transformers faz a inferência. Assim, os critérios de geração e o formato de chat permanecem iguais
entre os hubs. `trust_remote_code` fica desativado por padrão.

Ollama aplica o JSON Schema nativamente. Hugging Face e ModelScope recebem o schema como uma
instrução rígida no prompt e o JSON retornado ainda precisa passar pelo parser do extrator.

## API

```bash
pip install -e '.[api,documents]'
slm-kag serve --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H 'Content-Type: application/json' \
  -d '{"title":"Equipe","source":"crm://equipe","content":"Ana | trabalha_em | Acme"}'

curl -X POST http://localhost:8000/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Onde Ana trabalha?"}'
```

Upload direto de arquivo com metadados adicionais:

```bash
curl -X POST http://localhost:8000/v1/files \
  -F 'file=@relatorio.pdf' \
  -F 'source=ged://relatorios/2026' \
  -F 'metadata_json={"departamento":"financeiro","sigilo":"interno"}'
```

A documentação OpenAPI fica em `http://localhost:8000/docs`.

## Testes

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

## Configuração

Copie `.env.example` ou exporte as variáveis no processo. Os controles mais relevantes são:

| Variável | Padrão | Finalidade |
|---|---:|---|
| `SLM_KAG_DB_PATH` | `data/slm_kag.db` | arquivo SQLite |
| `SLM_KAG_DOCUMENT_LOADER` | `docling` | `docling` ou fallback explícito `text` |
| `SLM_KAG_MAX_FILE_SIZE` | `52428800` | limite do arquivo em bytes |
| `SLM_KAG_MAX_PAGES` | `500` | limite de páginas por documento |
| `SLM_KAG_EXTRACTOR` | `rules` | `rules`, `ollama`, `huggingface` ou `modelscope` |
| `SLM_KAG_GENERATOR` | `deterministic` | `deterministic`, `ollama`, `huggingface` ou `modelscope` |
| `SLM_KAG_OLLAMA_MODEL` | `qwen2.5:3b` | modelo local |
| `SLM_KAG_HF_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | ID ou diretório Hugging Face |
| `SLM_KAG_MODELSCOPE_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | ID ModelScope |
| `SLM_KAG_MODEL_DEVICE` | `auto` | `auto`, `cpu`, `cuda` ou `mps` |
| `SLM_KAG_MODEL_CACHE` | `data/models` | cache dos modelos |
| `SLM_KAG_MODEL_MAX_NEW_TOKENS` | `512` | limite de geração |
| `SLM_KAG_TRUST_REMOTE_CODE` | `false` | permite código remoto do repositório |
| `SLM_KAG_TOP_K` | `8` | máximo de evidências |
| `SLM_KAG_MAX_HOPS` | `2` | profundidade do subgrafo |
| `SLM_KAG_EVIDENCE_BUDGET` | `6000` | limite aproximado em caracteres |

Para incluir os dois hubs na imagem Docker:

```bash
SLM_KAG_EXTRAS=api,documents,huggingface,modelscope docker compose build
docker compose up api
```
