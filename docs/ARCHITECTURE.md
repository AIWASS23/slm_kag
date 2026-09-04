# Arquitetura do sistema SLM-KAG

## Objetivo

O sistema transforma documentos não estruturados em uma memória híbrida texto-grafo e entrega ao
SLM apenas um subgrafo pequeno, rastreável e relevante. A arquitetura adota o paradigma de
**pré-raciocínio simbólico/topológico** como caminho principal e permite o paradigma
**neuro-simbólico com ferramentas** pelos adaptadores de modelos locais. Isso reduz a carga de navegação e de
memória factual do modelo pequeno.

## Visão de contêineres e componentes

```mermaid
flowchart LR
    Browser[Navegador] --> UI[Streamlit / main.py]
    Client[CLI / cliente HTTP] --> API[FastAPI]
    UI --> IDX
    UI --> QA
    API --> IDX[IndexDocument]
    API --> QA[AnswerQuestion]

    subgraph Indexação
      FILE[PDF / Office / HTML / imagem] --> DOC[DoclingDocumentLoader]
      DOC -->|Markdown + metadata| IDX
      IDX --> CH[TextChunker]
      CH --> EX{FactExtractor}
      EX --> RULES[Regras / triplas explícitas]
      EX --> OEX[ChatModel + saída JSON]
    end

    subgraph Memória híbrida
      STORE[(SQLite)]
      FTS[FTS5 / chunks]
      KG[Property Graph]
      MI[entity_mentions\nindexação mútua]
      STORE --- FTS
      STORE --- KG
      STORE --- MI
    end

    IDX -->|transação única| STORE

    subgraph Raciocínio e resposta
      QA --> PLAN[LogicalPlanner]
      PLAN --> RET[HybridRetriever]
      RET --> FTS
      RET --> KG
      RET --> MI
      RET --> PPR[Personalized PageRank]
      PPR --> PRUNE[Ranking + evidence budget]
      PRUNE --> GEN{AnswerGenerator}
      GEN --> SAFE[Determinístico]
      GEN --> OGEN[ChatModel local]
      GEN --> VER[GroundingVerifier]
      VER -->|falha| SAFE
    end

    VER --> API

    subgraph Backends de modelo
      OLLAMA[Ollama HTTP]
      HF[Hugging Face Transformers]
      MS[ModelScope snapshot\n+ Transformers]
    end
    OEX --> OLLAMA
    OEX --> HF
    OEX --> MS
    OGEN --> OLLAMA
    OGEN --> HF
    OGEN --> MS
```

O núcleo depende de protocolos, não de FastAPI, SQLite, Ollama ou um hub específico. Isso permite trocar o banco por
Neo4j/OpenSPG/TuGraph e a busca textual por um índice vetorial sem alterar os casos de uso.

A GUI mantém somente configuração, uploads pendentes e histórico de conversa na sessão do
Streamlit. Documentos, chunks, grafo e metadados ficam no SQLite, portanto o usuário pode indexar
vários lotes antes de consultar e continuar acrescentando documentos depois. Cada upload recebe
uma origem estável derivada de SHA-256; reindexar o mesmo conteúdo/origem substitui sua versão em
vez de duplicá-la. Os binários temporários são apagados após a conversão pelo Docling.

## Implantação Kubernetes e entrega contínua

```mermaid
flowchart LR
    DEV[Push / Pull Request] --> CI[GitHub Actions]
    CI --> TEST[Testes + Ruff]
    CI --> VALIDATE[Kustomize + kubeconform]
    CI --> BUILD[Docker Buildx]
    BUILD --> GHCR[GitHub Container Registry]

    GUISVC[Service GUI :80] --> POD
    APISVC[Service API :8000] --> POD
    subgraph POD[Deployment slm-kag · 1 réplica · Recreate]
      GUI[Container Streamlit :8501]
      APIK[Container FastAPI :8000]
    end
    POD --> DATA[(PVC SQLite)]
    POD --> CACHE[(PVC cache de modelos)]
    OLLAMA[Overlay Ollama opcional] --> POD
```

GUI e API permanecem no mesmo Pod para que os dois processos compartilhem o filesystem e o lock
do SQLite. A estratégia `Recreate` encerra a versão anterior antes do novo Pod durante upgrades.
Escalar horizontalmente exige primeiro substituir `KnowledgeStore` por um armazenamento de rede;
por isso o manifesto fixa uma réplica. Hugging Face, ModelScope e Docling reutilizam um PVC de
cache, enquanto o overlay Ollama provisiona serviço e volume próprios.

Pull requests executam qualidade, validação Kubernetes e build sem publicação. Pushes aceitos e
tags publicam a imagem completa no GHCR com SBOM, metadados por branch/tag/SHA e atestado de
proveniência. Credenciais de cluster não fazem parte do CI: aplicar a imagem é uma decisão separada
e explícita do operador.

## Abstração de modelos

`ChatModel` define apenas `chat(messages, format_schema)`. Extração e geração dependem desse
contrato e não sabem onde os pesos foram obtidos:

| Backend | Aquisição | Execução | Restrição estruturada |
|---|---|---|---|
| Ollama | catálogo do runtime | API HTTP local `/api/chat` | JSON Schema nativo |
| Hugging Face | `from_pretrained(model_id)` | Transformers/PyTorch | schema no prompt + parse JSON |
| ModelScope | `snapshot_download(model_id)` | Transformers/PyTorch | schema no prompt + parse JSON |

Os clientes Transformers são lazy: construir a API não baixa pesos nem reserva GPU. O primeiro
pedido que realmente precise do SLM carrega tokenizer e modelo sob lock. O cache é configurável e
`trust_remote_code=false` é o padrão para reduzir a superfície de execução de código de terceiros.

São recomendados modelos *Instruct* que publiquem seu próprio `chat_template`. Para modelos base
sem template, o adaptador possui uma serialização textual conservadora como fallback.

## Fluxo de indexação

```mermaid
sequenceDiagram
    participant C as Cliente
    participant D as DoclingDocumentLoader
    participant I as IndexDocument
    participant X as FactExtractor
    participant S as KnowledgeStore

    C->>D: arquivo + source
    D->>D: stat, MIME e SHA-256
    D->>D: DocumentConverter.convert
    D->>D: export_to_markdown
    D->>I: conteúdo + file/docling metadata
    I->>I: ID estável pela source + chunks sobrepostos
    loop para cada chunk
      I->>X: texto do chunk
      X-->>I: triplas tipadas + confiança
    end
    I->>S: replace_document(document, chunks, facts)
    Note over S: substituição atômica da versão anterior
    S->>S: resolve entidades canônicas
    S->>S: grava relações e entity_mentions
    S->>S: atualiza FTS5 e remove órfãos
    S-->>C: contagens de chunks, fatos e entidades
```

O identificador do documento deriva de `source`, portanto uma nova ingestão da mesma origem
substitui a versão anterior. Relações, menções e chunks antigos são removidos por integridade
referencial; a atualização não deixa fatos obsoletos no grafo.

O arquivo é validado antes da conversão pelos limites de bytes e páginas. Uploads HTTP são gravados
em diretório temporário isolado e removidos ao fim da requisição. O banco retém o conteúdo
normalizado e os metadados, não o binário original.

## Fluxo de consulta

```mermaid
sequenceDiagram
    participant C as Cliente
    participant Q as AnswerQuestion
    participant R as HybridRetriever
    participant S as SQLite
    participant G as Gerador/SLM
    participant V as Verificador

    C->>Q: pergunta
    Q->>Q: compila QueryPlan
    par recuperação textual
      R->>S: FTS5(question)
    and recuperação topológica
      R->>S: entidades mencionadas
      R->>S: vizinhança até max_hops
      R->>R: Personalized PageRank
      R->>S: chunks ligados às entidades
    end
    R->>R: fusão, deduplicação e poda
    R-->>Q: E1..En com origem e score
    Q->>G: pergunta + evidências
    G-->>V: resposta com [E1..En]
    alt citações válidas
      V-->>C: answer + evidence + trace + grounded=true
    else sem suporte ou citação inválida
      Q->>Q: fallback determinístico
      Q-->>C: resposta segura ou grounded=false
    end
```

## Modelo de dados

```mermaid
erDiagram
    DOCUMENTS ||--o{ CHUNKS : contains
    CHUNKS ||--o{ ENTITY_MENTIONS : anchors
    ENTITIES ||--o{ ENTITY_MENTIONS : appears_in
    ENTITIES ||--o{ RELATIONS : source
    ENTITIES ||--o{ RELATIONS : target
    CHUNKS ||--o{ RELATIONS : proves

    DOCUMENTS {
      text id PK
      text source UK
      text title
      text content
      json metadata_json
    }
    CHUNKS {
      text id PK
      text document_id FK
      int position
      text content
    }
    ENTITIES {
      text id PK
      text normalized_name UK
      text name
      text kind
    }
    RELATIONS {
      text id PK
      text source_entity_id FK
      text predicate
      text target_entity_id FK
      real confidence
      text chunk_id FK
    }
    ENTITY_MENTIONS {
      text entity_id FK
      text chunk_id FK
      text role
      real confidence
    }
```

`ENTITY_MENTIONS` é a implementação da indexação mútua: qualquer entidade retorna ao fragmento
original e toda relação carrega o `chunk_id` que a sustenta. A API nunca precisa apresentar uma
tripla sem proveniência documental.

## Correspondência entre requisitos SOTA e implementação

| Risco/requisito do estudo | Decisão arquitetural |
|---|---|
| Gargalo de extração | schema nativo no Ollama; instrução de schema + parse estrito nos hubs; regras como baseline |
| Entidades inventadas | filtro exige sujeito e objeto ancorados ao texto normalizado do chunk |
| Fragilidade multi-hop | travessia feita pelo sistema e não pelo SLM; `max_hops` limita propagação |
| Distração por ruído | PageRank, fusão por ranking, `top_k` e orçamento de evidências |
| Lost in the Middle | contexto pequeno, relações linearizadas e evidências numeradas |
| Auditabilidade | aresta→chunk→documento, scores, plano e contagens no `trace` |
| Proveniência documental | Docling `origin` + MIME, SHA-256, páginas, status e metadados do usuário |
| Privacidade | SQLite e Ollama locais; nenhuma chamada de nuvem no caminho padrão |
| Alucinação | resposta exige citações existentes; fallback seguro quando a validação falha |

## Limites conscientes e evolução

Esta primeira versão é uma vertical slice funcional, adequada a um nó e a corpora pequenos ou
médios. A evolução para produção deve preservar as portas em `ports.py` e trocar adaptadores:

1. **Escala de busca:** adicionar embeddings locais e Qdrant/pgvector; manter FTS para recuperação
   lexical e aplicar Reciprocal Rank Fusion.
2. **Escala do grafo:** implementar `KnowledgeStore` sobre OpenSPG, Neo4j ou TuGraph e executar PPR
   próximo aos dados.
3. **Esquema de domínio:** validar tipos e predicados contra uma ontologia versionada antes do
   commit, com fila de quarentena para fatos de baixa confiança.
4. **Raciocínio formal:** estender `QueryPlan` para filtros temporais, aritmética, agregações e regras
   de negócio executadas por operadores, nunca por texto livre.
5. **Qualidade:** adicionar golden sets de perguntas, métricas Recall@K, MRR, exact match, fidelidade
   de citações, latência p95 e taxa de fallback por modelo.
6. **Operação:** autenticação, limites de tamanho, logs estruturados sem conteúdo sensível,
   OpenTelemetry, snapshots e política de retenção.
7. **Destilação:** registrar planos e evidências verificadas como dataset offline para DRAG/KARD,
   mantendo treinamento fora do caminho de inferência.

## Estrutura do código

```text
src/slm_kag/
├── domain/models.py              # entidades, evidências, planos e respostas
├── ports.py                      # contratos de armazenamento, extração e geração
├── application/indexing.py       # chunking e caso de uso de ingestão
├── application/querying.py       # planner, PPR, fusão, verificação e resposta
├── infrastructure/sqlite_store.py# grafo, FTS5 e mutual indexing
├── infrastructure/document_loaders.py # Docling e metadados dos arquivos
├── infrastructure/extractors.py # regras e extração estruturada agnóstica
├── infrastructure/ollama.py      # cliente local e gerador fundamentado
├── infrastructure/transformers_client.py # Hugging Face e ModelScope
├── bootstrap.py                  # composição de dependências
├── api.py                        # adaptador HTTP
├── cli.py                        # adaptador de linha de comando
└── streamlit_app.py              # upload em lotes, configuração e chat fundamentado
main.py                           # ponto de entrada da GUI Streamlit
```
