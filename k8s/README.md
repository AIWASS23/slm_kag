# Kubernetes

Os manifests usam Kustomize, já incorporado ao `kubectl`. GUI e API executam como dois containers
do mesmo Pod porque ambos acessam o SQLite em `/app/data/slm_kag.db`. O Deployment mantém uma
única réplica e usa estratégia `Recreate`, evitando duas instâncias concorrentes durante upgrades.

## Escolher e configurar a imagem

O CI publica a imagem completa em `ghcr.io/<proprietário>/<repositório>`. Antes do primeiro deploy,
edite `newName` no `kustomization.yaml` do overlay escolhido. Exemplo:

```yaml
images:
  - name: slm-kag
    newName: ghcr.io/marcelo/slm-kag
    newTag: latest
```

Se o pacote GHCR for privado, crie um `imagePullSecret` no namespace e acrescente seu nome em
`spec.template.spec.imagePullSecrets` no Deployment. Nenhuma credencial é versionada neste projeto.

## Variantes

| Overlay | Extração | Geração | Recurso adicional |
|---|---|---|---|
| `default` | regras | determinística | nenhum |
| `huggingface` | Hugging Face | Hugging Face | cache persistente de modelos |
| `modelscope` | ModelScope | ModelScope | cache persistente de modelos |
| `ollama` | Ollama | Ollama | Ollama, PVC e download inicial do Qwen |

Os overlays Hugging Face e ModelScope usam o Qwen2.5-0.5B para caber em um cluster de teste. Altere
o ID no respectivo `model-config.yaml` e ajuste memória/GPU em `base/deployment.yaml` para modelos
maiores.

## Aplicar

Para o modo offline padrão:

```bash
kubectl kustomize k8s/overlays/default
kubectl apply -k k8s/overlays/default
kubectl -n slm-kag rollout status deployment/slm-kag
```

Troque `default` por `huggingface`, `modelscope` ou `ollama` para usar outro backend. A primeira
inicialização de um backend de modelo pode demorar enquanto os pesos são gravados no PVC.

Os Services são `ClusterIP`. Para acesso local:

```bash
kubectl -n slm-kag port-forward service/slm-kag-gui 8501:80
kubectl -n slm-kag port-forward service/slm-kag-api 8000:8000
```

A GUI estará em `http://localhost:8501`, a API em `http://localhost:8000` e o OpenAPI em
`http://localhost:8000/docs`.

## Configuração e armazenamento

- `base/configmap.yaml`: limites, modelos e backends.
- `slm-kag-data`: 10 Gi para SQLite e documentos normalizados.
- `slm-kag-model-cache`: 30 Gi para Hugging Face, ModelScope, Torch e Docling.
- `ollama-models`: 30 Gi, criado apenas pelo overlay Ollama.

Os PVCs não fixam `storageClassName`; o StorageClass padrão do cluster será usado. Ajuste tamanho,
classe e recursos antes de produção. Os containers da aplicação executam sem root, sem token da
ServiceAccount, com filesystem raiz somente leitura, capabilities removidas e probes HTTP.

Não escale `deployment/slm-kag` enquanto o adaptador de persistência for SQLite. Para alta
disponibilidade horizontal, implemente `KnowledgeStore` sobre um banco compartilhado antes de
aumentar `replicas`.

## Diagnóstico e remoção

```bash
kubectl -n slm-kag get pods,services,pvc
kubectl -n slm-kag logs deployment/slm-kag -c gui
kubectl -n slm-kag logs deployment/slm-kag -c api
kubectl delete -k k8s/overlays/default
```

Apagar os manifests não necessariamente remove os volumes provisionados. Confira a política de
retenção do StorageClass antes de excluir PVCs.

