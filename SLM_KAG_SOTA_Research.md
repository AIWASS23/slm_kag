# SLMs Aplicadas a KAG (Knowledge-Augmented Generation): Estado da Arte, Arquiteturas e Pesquisa SOTA

> **Autor / Pesquisa:** Compilação do Estado da Arte (SOTA) em Modelos de Linguagem Pequenos (SLMs) integrados a Grafos e Geração Aumentada por Conhecimento (KAG).  
> **Data:** Setembro de 2026  
> **Escopo:** Artigos fundamentais (2023–2026), taxonomia técnica, benchmarks, gargalos estruturais e diretrizes de implementação.

---

## 1. Visão Geral e Motivação

### 1.1 O Limite do RAG Tradicional e a Emergência do KAG
O **RAG convencional (Retrieval-Augmented Generation)**, baseado em busca vetorial densa sobre pedaços de texto (*chunks*), revolucionou a aplicação de Inteligência Artificial generativa. No entanto, ele sofre de severas falhas estruturais em cenários profissionais e de raciocínio complexo:
* **Incapacidade de raciocínio multi-hop:** Dificuldade em correlacionar entidades e relações espalhadas por múltiplos documentos.
* **Miopia semântica por similaridade de cosseno:** Similaridade vetorial não implica pertinência lógica ou causalidade.
* **Insensibilidade a regras formais e temporais:** Fraqueza ao lidar com restrições numéricas, condições booleanas, sequências temporais e conformidade regulatória.
* **Ruído do GraphRAG ingênuo (OpenIE):** Abordagens que extraem triplas livres de texto geram grafos ruidosos, com entidades duplicadas, relações vagas e custo computacional astronômico de indexação.

O framework **KAG (Knowledge-Augmented Generation)** surge para solucionar essas lacunas através da integração bidirecional entre **Grafos de Conhecimento Estruturados (Knowledge Graphs / SPG - Semantic Property Graph)**, representações lógicas e modelos neurais de linguagem.

```
       +-------------------------------------------------------------------------+
       |                           KAG ARCHITECTURE                              |
       +-------------------------------------------------------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                                                                   |
        v                                                                   v
+-----------------------+                                         +-----------------------+
|  Structured Knowledge |                                         |   Unstructured Data   |
| (Domain SPG / Triples)|                                         | (Document Chunks/Txt) |
+-----------------------+                                         +-----------------------+
        |                                                                   |
        +-----------------> [ Mutual-Indexing Engine ] <--------------------+
                                    |
                                    v
                     +-----------------------------+
                     |  Logical Form Hybrid Solver |
                     +-----------------------------+
                                    |
             +----------------------+----------------------+
             |                                             |
             v                                             v
+--------------------------+                 +--------------------------+
|  Graph/Symbolic Engine   |                 |   Small Language Model   |
| (PPR, GNN, Subgraph/SPG) |                 |    (1B - 8B SLM Agent)   |
+--------------------------+                 +--------------------------+
             |                                             |
             +----------------------+----------------------+
                                    |
                                    v
                    [ Verified, Explainable Output ]
```

### 1.2 Por que SLMs (*Small Language Models*) em KAG?
Modelos como **Llama 3.2 (1B/3B)**, **Gemma 2/3 (1B/2B/4B/9B)**, **Qwen 2.5 (0.5B/1.5B/3B/7B)** e **Phi-3/Phi-4 (3.8B/14B)** tornaram-se alternativas cruciais devido a:
1. **Soberania de dados e privacidade:** Execução *on-premises* ou *on-device* (Edge Computing) sem vazamento de dados confidenciais (saúde, financeiro, governamental).
2. **Eficiência de custos e latência:** Custo de inferência até 95% menor e tempos de resposta na ordem de milissegundos.
3. **Pegada computacional e consumo energético:** Viabilidade em servidores modestos sem a necessidade de clusters de GPUs de altíssimo custo (A100/H100).

### 1.3 O Casamento SLM + KAG: Sinergia Neuro-Simbólica
Modelos pequenos possuem menor capacidade de memorização paramétrica e são propensos a falhas de raciocínio dedutivo complexo. Em contrapartida, o KAG oferece a âncora factual exata que o SLM necessita:
* **Terceirização da memória factual:** A base de conhecimento não precisa ser memorizada nos pesos do SLM; ela reside no grafo de conhecimento.
* **Restrição do espaço de busca:** O motor simbólico (SPG/GNN/PageRank) faz a travessia no grafo e fornece subgrafos podados, permitindo que o SLM atue apenas como tradutor semântico e sumarizador, sem alucinações.
* **Redução drástica de contexto:** O SLM recebe triplas lógicas condensadas em vez de dezenas de páginas de texto cru ruidoso.

---

## 2. Artigos do Estado da Arte (SOTA)

Abaixo estão os trabalhos científicos mais impactantes que fundamentam a interseção entre SLMs e KAG / Grafos de Conhecimento:

| Artigo | Autores / Instituições | Ano | Contribuição Central | Benchmarks / Resultados |
| :--- | :--- | :--- | :--- | :--- |
| **KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation** | Lei Liang et al. (Ant Group / OpenSPG) | 2024–2025 | Framework pioneiro de KAG integrando SPG (Semantic Property Graph), indexação mútua e raciocínio híbrido guiado por forma lógica. | +33.5% F1 no HotpotQA; +19.6% F1 no 2WikiMultiHopQA; aplicado em E-Gov e E-Health. |
| **MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation** | Tianyu Fan, Jingyuan Wang, Xubin Ren, Chao Huang (HKU) | 2025 | RAG baseado em grafos projetado especificamente para SLMs. Elimina a dependência de capacidades semânticas avançadas via indexação heterogênea. | Performance equivalente a LLMs em SLMs usando apenas **25% do armazenamento**; benchmark *LiHua-World*. |
| **DRAG: Distilling RAG for SLMs from LLMs via Evidence and Graph Distillation** | Jennifer Chen, Aidar Myrzakhan et al. | 2025 | Destilação de capacidades de recuperação e raciocínio de LLMs para SLMs ancorada em grafos de conhecimento e evidências ranqueadas. | Supera o MiniRAG em até **27.7%** nos mesmos modelos leves, preservando privacidade local. |
| **Enhancing Small Language Models Reasoning through Knowledge Graph Grounding** | Dimitrios Kelesis, Konstantinos Bougiatiotis, Georgios Paliouras | 2026 | Framework neuro-simbólico que transforma SLMs (Gemma 3 1B/4B, Llama 3.2 3B) em agentes com chamadas de ferramenta especializadas (`extract_facts`, `get_hint` via RGCN). | Ganho de **1.5x a 2x** em raciocínio relacional no benchmark CLUTRR; diagnóstico de fragilidades dedutivas sequenciais. |
| **The Role of Exploration Modules in SLMs for KGQA (SLM-ToG)** | Yi-Jie Cheng, Oscar Chew, Yun-Nung Chen | 2025 | Desacopla o raciocínio em grafos: delega a exploração/travessia do grafo a um módulo leve dedicado, aliviando o SLM para a síntese final. | Supera o ToG (*Think-on-Graph*) tradicional em SLMs, removendo o gargalo de navegação do modelo. |
| **KARD: Knowledge-Augmented Reasoning Distillation for SLMs** | Minki Kang et al. (KAIST) - *NeurIPS* | 2023 | Destilação de cadeias de raciocínio (rationales) aumentadas por recuperação externa de conhecimento para SLMs. | Modelos T5 de **250M** superam modelos de **3B (12x maiores)** no MedQA-USMLE e StrategyQA. |
| **G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding** | Xiaoxin He et al. (Notre Dame) - *NeurIPS* | 2024 | Recuperação sobre grafos textuais formulada como o problema de otimização *Prize-Collecting Steiner Tree (PCST)* com *soft prompting*. | Elimina alucinação em QA estruturado; escalável para grafos que excedem a janela de contexto. |
| **HippoRAG & LightRAG** | Gutiérrez et al. (OSU) / Guo et al. (HKU) | 2024–2025 | *HippoRAG:* Memória associativa inspirada no hipocampo via Personalized PageRank (PPR); *LightRAG:* Recuperação em dois níveis (entidades/relações + temas globais). | HippoRAG: 10-30x mais barato e 6-13x mais rápido que buscas iterativas; LightRAG viabiliza SLMs locais (Phi-4, Qwen 2.5). |

---

## 3. Análise Detalhada dos Trabalhos Centrais

### 3.1 Framework KAG (Ant Group / OpenSPG)
* **Referência:** Liang et al., *KAG: Boosting LLMs in Professional Domains via Knowledge Augmented Generation* (arXiv:2409.13731 / ACM KDD 2025).
* **Estrutura Técnica:**
  1. **LLMFriSPG (*LLM-Friendly Semantic Property Graph*):** Diferente de grafos RDF rígidos ou OpenIE caótico, o SPG organiza conhecimento na hierarquia DIKW (*Data, Information, Knowledge, Wisdom*), acomodando tanto extrações livres quanto restrições de esquemas de domínio.
  2. **Mutual Indexing:** Cria um elo de mão dupla entre entidades/relações do grafo e os fragmentos de texto originais (*chunks*), viabilizando rastreabilidade bidirecional.
  3. **Logical-Form-Guided Hybrid Reasoning Engine:** Em vez de depender apenas de *prompts* livres, o motor decompõe a pergunta do usuário em uma forma lógica abstrata (expressão simbólica/DSL), executando uma mistura planejada de:
     * Travessia no Grafo de Conhecimento (SPG Query)
     * Recuperação vetorial densa
     * Operações matemáticas e regras de negócio
  4. **KAG-Thinker:** Módulo especializado em decomposição iterativa em largura e profundidade com mitigação de ruído.
* **Aplicação com SLMs:** Através das interfaces nativas do OpenSPG (suporte a clientes Ollama, vLLM e SGLang), SLMs como **Qwen 2.5 7B** e **Llama 3.2 3B** podem ser acoplados como motores executores do *kg-solver* e extratores no *kg-builder*.

---

### 3.2 MiniRAG: RAG Hiper-Eficiente Dedicado a SLMs
* **Referência:** Fan et al., *MiniRAG: Towards Extremely Simple Retrieval-Augmented Generation* (arXiv:2501.06713, 2025).
* **O Problema Resolvido:** Modelos pequenos degradam rapidamente em GraphRAGs tradicionais porque estes exigem que o LLM entenda resumos de nós hiperdensos e realize deduções semânticas além de sua capacidade paramétrica.
* **Inovações Principais:**
  1. **Semantic-aware Heterogeneous Graph Indexing:** Unifica blocos de texto (*chunks*) e entidades nomeadas em uma única estrutura de grafo heterogêneo sem exigir ontologias complexas no momento da indexação.
  2. **Lightweight Topology-Enhanced Retrieval:** A recuperação explora a topologia do grafo para conectar entidades sem demandar que o SLM faça inferências semânticas pesadas.
  3. **Resultados:** Redução de 75% no espaço em disco e memória em comparação aos concorrentes, permitindo que SLMs rodem em dispositivos periféricos (*Edge Devices*) com o benchmark *LiHua-World*.

---

### 3.3 DRAG: Destilação de KAG/RAG para SLMs
* **Referência:** Chen et al., *DRAG: Distilling RAG for SLMs from LLMs to Transfer Knowledge and Mitigate Hallucination via Evidence and Graph-based Distillation* (arXiv:2506.01954, 2025).
* **Mecanismo:**
  * O conhecimento estrutural de um grafo de conhecimento associado a um LLM de grande porte (como Llama 3 70B ou GPT-4) é destilado para um SLM através de:
    * **Graph-guided distillation:** O SLM aprende a prever probabilidades de nós e relações baseadas na topologia do grafo.
    * **Ranked Evidence distillation:** O SLM é treinado para ignorar documentos distratores e focar na evidência essencial.
  * **Resultados:** Ganho de **27.7% de acurácia** sobre o MiniRAG ao utilizar modelos SLM equivalentes, minimizando vazamento de dados de privacidade (*Privacy-preserving RAG*).

---

### 3.4 SLM Neuro-Simbólico com RGCN (Kelesis et al., 2026)
* **Referência:** Kelesis et al., *Enhancing Small Language Models Reasoning through Knowledge Graph Grounding* (arXiv:2607.14149, 2026).
* **Arquitetura Agêntica Minimalista:**
  * Avaliado em **Gemma 3 (1B, 4B)** e **Llama 3.2 (3B)**.
  * O SLM opera com duas ferramentas atômicas:
    1. `extract_facts`: Extração simbólica de triplas `(sujeito, predicado, objeto)` a partir do texto.
    2. `get_hint`: Envia o subgrafo extraído para uma **RGCN (Relational Graph Convolutional Network)**, que calcula a inferência dedutiva e devolve uma "dica simbólica" (*expert hint*) para o SLM.
* **Descobertas Críticas:**
  * **Ganho:** Melhora de 1.5x a 2x no raciocínio relacional (CLUTRR).
  * **Gargalo de Extração (*Extraction Bottleneck*):** Se o SLM falhar na extração inicial de uma única tripla, os erros se propagam em cascata (*Sequential Deductive Fragility*).
  * **Efeito Distração (*Distraction Effect*):** Modelos ultra-pequenos (1B) tendem a se confundir quando triplas ruidosas geradas por eles mesmos são injetadas de volta no contexto, mesmo que a dica da RGCN esteja correta.

---

### 3.5 SLM-ToG: Módulos de Exploração Desacoplados
* **Referência:** Cheng, Chew & Chen, *The Role of Exploration Modules in Small Language Models for Knowledge Graph Question Answering* (arXiv:2509.07399, 2025).
* **Conceito:** No framework *Think-on-Graph (ToG)* original, o LLM decide interativamente quais nós do grafo explorar. SLMs falham dramaticamente nessa tarefa por perderem o controle do estado e alucinarem arestas inexistentes.
* **Solução:** Substituir a navegação do SLM por um **módulo de exploração determinístico e leve** (baseado em heurísticas de conectividade e similaridade de embeddings locais). O SLM é invocado apenas para validação e resposta final.

---

## 4. Paradigmas de Arquitetura para SLMs em KAG

Podemos classificar as abordagens atuais em três paradigmas arquiteturais principais:

```
PARADIGMA 1: Pré-Raciocínio Simbólico / Topológico
[Pergunta] ---> [Motor do Grafo (PPR / Steiner Tree / SPG)] ---> [Subgrafo Podado] ---> [SLM Gerador] ---> [Resposta]

PARADIGMA 2: Agente Neuro-Simbólico com Ferramentas
                +-------------------+
                |     SLM Agent     |
                +-------------------+
                  | ^           | ^
   extract_facts  | | triplas   | | get_hint
                  v |           v |
            [Grafo/Parser]    [GNN/RGCN Engine]

PARADIGMA 3: Destilação de Raciocínio KAG (Off-line / Fine-Tuning)
[Teacher LLM + KG] ---> [Destilação de Trajetórias Lógicas + Evidências] ---> [SLM Especialista em KAG]
```

### Paradigma 1: Pré-Raciocínio Simbólico / Topológico
* **Representantes:** G-Retriever, HippoRAG, MiniRAG.
* **Funcionamento:** O SLM não toca no grafo durante a busca. Algoritmos matemáticos (Personalized PageRank, Prize-Collecting Steiner Tree) operam sobre o grafo e entregam um subgrafo limpo e conciso ao SLM.
* **Vantagem para SLMs:** Máxima estabilidade. O SLM atua na sua zona de conforto: interpretação de contexto limpo e síntese linguística.

### Paradigma 2: Agente Neuro-Simbólico com Ferramentas
* **Representantes:** OpenSPG KAG-Thinker, Kelesis et al. (2026).
* **Funcionamento:** O SLM emite chamadas de função lógicas (`tool calls`) formatadas rigidamente (JSON/DSL), e microsserviços simbólicos executam consultas em bancos de grafos (Neo4j, TuGraph, OpenSPG) ou modelos de aprendizado geométrico (GNNs).
* **Vantagem para SLMs:** Capacidade de orquestração flexível e auditabilidade total de cada passo do raciocínio.

### Paradigma 3: Destilação de Raciocínio KAG
* **Representantes:** KARD, DRAG.
* **Funcionamento:** Treinamento ou fine-tuning de SLMs usando conjuntos de dados gerados por modelos maiores integrados a grafos.
* **Vantagem para SLMs:** Permite que modelos sub-3B superem modelos 10x maiores sem overhead excessivo em tempo de inferência.

---

## 5. Benchmarks e Métricas Relevantes

Para avaliar KAG com SLMs, a literatura consolidou os seguintes benchmarks:

1. **LiHua-World (Fan et al., 2025):** Criado especificamente para avaliar RAG leve e em dispositivos periféricos com consultas complexas e restrições severas de hardware.
2. **CLUTRR (Kinship Reasoning):** Teste rigoroso de raciocínio indutivo e dedutivo sobre árvores genealógicas; avalia saltos relacionais de 2 a 10 passos (*multi-hop reasoning*).
3. **HotpotQA & 2WikiMultiHopQA:** Padrões de referência para perguntas que exigem conexão de múltiplos fatos dispersos em várias páginas ou entidades.
4. **MedQA-USMLE & StrategyQA:** Testes de conhecimento de domínio crítico (medicina e raciocínio de bom senso avançado).
5. **GraphQA (Xia et al., 2024):** Benchmark diversificado cobrindo raciocínio em grafos textuais, compreensão de grafos de cena e senso comum.

---

## 6. Desafios e Gargalos Conhecidos de SLMs em KAG

Apesar do imenso potencial, a literatura recente identifica 4 desafios cruciais:

1. **Gargalo de Extração de Informação (*Extraction Bottleneck*):**
   * SLMs menores que 3B sofrem para converter texto livre em triplas `(head, relation, tail)` perfeitamente alinhadas com um esquema sem alucinar relações fantasmas.
   * *Mitigação SOTA:* Uso de esquemas pré-definidos via SPG (como no OpenSPG) e decodificação guiada por gramática (*grammar-constrained decoding* via outlines, instructor ou JSON schemas).
2. **Fragilidade Dedutiva Sequencial (*Sequential Deductive Fragility*):**
   * Em cadeias multi-hop, um único nó intermediário incorreto gera um efeito dominó de falhas dedutivas que o SLM não consegue auto-corrigir sem auxílio externo.
   * *Mitigação SOTA:* Verificadores externos simbólicos a cada passo de raciocínio.
3. **Sensibilidade ao Ruído / Efeito Distração (*Distraction Effect*):**
   * SLMs perdem a coerência muito mais rápido que modelos de 70B quando deparados com triplas irrelevantes ou falsos positivos retornados na busca.
   * *Mitigação SOTA:* Algoritmos de poda estrita (*Steiner Tree pruning*, ranqueamento de evidências do DRAG).
4. **Tamanho da Janela de Contexto e Atenção:**
   * Embora modelos recentes tenham contextos nominais de 32k a 128k tokens, o fenômeno *"Lost in the Middle"* é acentuado em SLMs. A representação via grafo precisa ser ultra-densa.

---

## 7. Guia Prático: Como Implementar KAG com SLMs Hoje

Para engenheiros e pesquisadores que desejam construir um sistema KAG com SLMs locais:

```
[Documentos / Ontologia]
           │
           ▼
[OpenSPG / LightRAG]  <─── Geração de Índices / Subgrafos
           │
           ▼
 [FastAPI / Ollama]   <─── Qwen 2.5 (3B / 7B) ou Llama 3.2 (3B)
           │
           ▼
   [Grammar Engine]   <─── Outlines / Instructor (JSON / Tool Calling)
           │
           ▼
[Resposta Estruturada e Factual]
```

### 7.1 Seleção de Modelos SLM Recomendados (Estado da Arte)
* **Qwen 2.5 (3B / 7B - Instruct):** Excelente capacidade de seguir instruções em formatos estruturados (JSON/YAML), suporte a múltiplos idiomas e grande precisão sintática em *tool calling*.
* **Llama 3.2 (3B - Instruct):** Alta eficiência em tarefas de raciocínio lógico e compatibilidade universal com runtimes (vLLM, Ollama, ONNX Runtime Edge).
* **Gemma 3 (1B / 4B):** Novo estado da arte para raciocínio compacto e extração simbólica quando acoplado a GNNs.
* **Phi-4 (14B) / Phi-3.5-mini (3.8B):** Raciocínio matemático e lógico comparável a modelos significativamente maiores devido à qualidade dos dados sintéticos de treinamento.

### 7.2 Frameworks de KAG / Graph-RAG Recomendados
* **OpenSPG / KAG (Ant Group):** [GitHub OpenSPG/KAG](https://github.com/OpenSPG/KAG) — A biblioteca de código aberto mais alinhada formalmente com a arquitetura KAG profissional, suporte a SPG, indexação mútua e conectores com Ollama/vLLM.
* **LightRAG (HKUDS):** [GitHub HKUDS/LightRAG](https://github.com/HKUDS/LightRAG) — Extremamente rápido, ideal para implantação com SLMs locais com indexação em dois níveis.
* **MiniRAG (HKUDS):** [GitHub HKUDS/MiniRAG](https://github.com/HKUDS/MiniRAG) — Escolha ideal para cenários *on-device* com hardware estritamente limitado.
* **HippoRAG (OSU-NLP):** [GitHub OSU-NLP-Group/HippoRAG](https://github.com/OSU-NLP-Group/HippoRAG) — Excelente quando a tarefa principal envolve busca associativa e raciocínio multi-hop via PageRank.

---

## 8. Conclusão e Tendências Futuras

A combinação de **SLMs com KAG** representa a maturidade da IA generativa pragmática:
* A dependência de modelos gigantescos em nuvem (*cloud-based LLM APIs*) está cedendo espaço para **sistemas híbridos e neuro-simbólicos**, onde modelos pequenos atuam em perfeita harmonia com estruturas determinísticas de dados.
* A evolução do KAG consolida o princípio de que **"representação de conhecimento estruturada supera o aumento indiscriminado de parâmetros"** quando o objetivo é precisão factual, custo acessível e auditabilidade.
* Os próximos anos verão a padronização de agentes cooperativos SLM-KAG na borda (*edge*), permitindo que dispositivos pessoais, sistemas hospitalares e infraestruturas críticas operem com IA avançada sem alucinações e em total conformidade com requisitos de privacidade.

---
*Documento gerado como consolidação de literatura científica SOTA para pesquisa e engenharia de sistemas KAG.*
