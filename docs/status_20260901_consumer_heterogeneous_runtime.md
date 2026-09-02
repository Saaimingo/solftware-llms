# Checkpoint — Consumer Heterogeneous Runtime

Data do checkpoint: 2026-09-01

Este documento preserva o estado científico e operacional do projeto antes da abertura de um novo bloco/thread. Ele foi criado no GitHub porque o thread atual do Hermes ficou preso em `Summarizing thread` e a branch local `research/consumer-heterogeneous-runtime` ainda não estava sincronizada com o remoto.

> Importante: este checkpoint documenta o estado conhecido e validado. Ele NÃO substitui os artefatos locais, binários, traces ou commits que ainda só existem no PC. O branch remoto deste checkpoint parte de `master` e serve como ponto de recuperação/documentação, não como substituto da branch local ainda não publicada.

## 1. Objetivo central atual

O projeto evoluiu de um simples otimizador de VRAM para uma investigação mais ampla:

**Qual é a melhor arquitetura de inferência de LLM para hardware consumer discreto composto por CPU + RAM + PCIe + GPU + VRAM limitada?**

Plataforma de referência:

- Windows 11 x64
- Ryzen 7 5700
- ~40 GB RAM
- PCIe 3.0 x16
- NVIDIA RTX 3060 12 GB
- CUDA Toolkit 12.4
- MSVC 19.44
- CMake 4.4.3

Modelo principal atual:

`D:\Models\Qwen3.6-35B-A3B-Q2_K.gguf`

Características já observadas/documentadas:

- 40 layers
- 256 experts roteados por layer
- top-8 experts por decisão
- modelo MoE
- execução local em RTX 3060 12 GB

## 2. Hipótese arquitetural maior

A direção atual não é tratar a RTX 3060 como uma “H100 pequena”.

A hipótese é criar, primeiro no runtime e possivelmente no futuro também na própria arquitetura do modelo, uma execução nativa para:

CPU + RAM + PCIe + GPU + VRAM

A ideia é tratar esses componentes como uma única máquina heterogênea, com diferentes níveis de velocidade, capacidade e latência.

Princípios em investigação:

- evitar movimento de dados desnecessário;
- antecipar movimento inevitável;
- sobrepor transferência com compute;
- reduzir bytes transferidos;
- reorganizar layout/representação;
- usar CPU deliberadamente como parte do sistema;
- usar VRAM como cache hierárquico;
- model-aware runtime;
- hardware-aware routing no futuro, se a evidência justificar.

## 3. Fronts de pesquisa existentes

### Front 1 — Adaptive Runtime / RAM↔VRAM / expert residency

Conceito já desenvolvido:

- COLD = RAM/mmap
- WARM = RAM pinned/preparada
- STANDBY = VRAM especulativa
- HOT = VRAM confirmada/em uso

Objetivo: reduzir GPU idle causado por experts que precisam atravessar PCIe.

### Front 2 — Representation Search

Objetivo: encontrar representações melhores do que simplesmente descer Q4→Q3→Q2, preservando mais qualidade por byte e, no futuro, talvez favorecendo trânsito RAM→PCIe→GPU.

O experimento de QRM pós-quantização low-rank/sparse/híbrido terminou em NO-GO inicial sob os controles usados.

### Front 3 — External Capability Memory

Hipótese futura: manter backbone saudável e adicionar memória/capacidade externa especializada, sem obrigar todo conhecimento/capacidade a residir na VRAM.

### Front 4 — Adaptive Speculative Decoding

Linha inspirada por trabalhos como AngelSpec/DFly/DSpark, mas com foco consumer single-user/batch 1. Ainda não implementada.

## 4. Router tracing real — estado validado

Foi instrumentada uma revisão experimental do `llama.cpp` para capturar decisões reais do router MoE.

Estado científico anterior consolidado:

- 18/18 traces válidos
- 81.560 eventos reais
- 6 categorias × 3 sessões
- `provenance=real_router_trace` em 100%
- 0 JSON inválido
- 40 layers
- 8 experts/evento

Distribuição por workload:

- A = 15.160
- B = 12.720
- C = 12.920
- D = 16.440
- E = 12.920
- F = 11.400

Baselines observados:

- frequent ≈ 0,5%
- previous ≈ 0%
- markov ≈ 5,0% session-to-session
- markov ≈ 5,8% cross-workload
- ngram2 ≈ 3,5–4,4%
- oracle bundle-exato ≈ 8,4–8,7%
- expert-level markov ≈ 10–11% precision/recall

Hierarchical offline:

- train = 57.092 eventos
- validation = 24.468 eventos
- top1_hit ≈ 27,4%
- bundle_hit ≈ 9,6%
- baseline gpu_idle_estimated ≈ 299,5 s
- 64 MiB STANDBY ≈ 286,3 s
- 256 MiB STANDBY ≈ 267,4 s
- redução offline estimada ≈ 10%
- plateau próximo de 256 MiB no modelo antigo

Classificação científica atual da linha de trace/predição:

`GO_PARTIAL_REAL_TRACE`

Significado estrito:

- existe sinal temporal real;
- existe benefício offline plausível;
- NÃO existe ainda prova de ganho real em tok/s;
- NÃO existe autorização para cache de produção.

## 5. Artefatos finais da bateria anterior

Foram gerados localmente:

- `artifacts/analysis/20260831-battery-final-report.md`
  - SHA-256: `4abdc409c3bb16928eb6da3d149c1e4f70078394c09c482d1c55fd42952982ad`
- `artifacts/analysis/battery_analysis_full.json`
  - SHA-256: `8423cc8d3283265c6c0377d92f8cb371e259135eb6773b73fab109f766275c72`
- `artifacts/analysis/battery_manifest.json`
  - SHA-256: `409ad3a0943895837ddea81b3aee466f62b55892149a7e43387e23e90bab2ca9`

Identidades históricas preservadas:

- branch experimental anterior: `experiment/real-router-trace`
- project HEAD anterior: `44e19a0d3b07f4ad32bbc19aca5706b93955f4cc`
- vendor commit: `9c0dc486bfc332e497b6c66edf0fe1970fcc09ae`
- vendor base: `50f068ffffc3e0e4c9c2e4139281c6075224f429`
- patch SHA-256: `9b8f359dbe4e79bae8a2dd3ef729d580cbd26820bccaae49182143d1f93d8139`
- binary anterior SHA-256: `43fcd9033d752aea9908875d3f9b60e10c6dc6746fa143bc09cf448da0f304aa`
- model SHA-256: `f232a8f183e557860f335203d25cbaa6a65dddfde68bf27621e2fbdc1d9e81d7`

## 6. Mudança de eixo: famílias de experts

Durante a investigação surgiu uma hipótese mais forte do que tentar adivinhar o bundle exato cedo demais:

**Talvez seja possível prever uma família/região de experts antes de saber o expert exato.**

A ideia é:

hidden state parcial / sinal antecipado
→ família provável
→ preparar candidatos
→ router definitivo
→ confirmar quais top-8 realmente serão usados

Métrica central futura:

`REAL_SELECTED_EXPERT_COVERAGE`

Ou seja: dos 8 experts realmente escolhidos pelo router, quantos já estavam dentro do conjunto/família previamente preparada?

Também surgiu a hipótese de lookahead multi-layer:

- A = próxima layer, mais quente
- B = próxima+1
- C = próxima+2
- D = próxima+3

A quantidade de candidatos pode variar conforme confiança e distância temporal.

## 7. Descoberta factual importante — tamanho real de expert

Foi feita medição direta no GGUF real.

Artefato local:

`artifacts/analysis/expert_size_real.json`

Resultado:

- 40 layers
- 256 experts/layer
- **1.138.688 bytes por expert**
- valor idêntico nas 40 layers observadas

Composição de um expert:

- `ffn_down_exps.weight` — Q3_K — 450.560 bytes
- `ffn_gate_exps.weight` — Q2_K — 344.064 bytes
- `ffn_up_exps.weight` — Q2_K — 344.064 bytes
- total = **1.138.688 bytes**

Isso invalida o antigo `expert_bytes=20 MiB` como dimensão factual. O valor de 20 MiB deve permanecer apenas como parâmetro histórico de uma simulação anterior.

Capacidade bruta aproximada por orçamento:

- 32 MiB → 29 experts
- 64 MiB → 58 experts
- 128 MiB → 117 experts
- 256 MiB → 235 experts
- 512 MiB → limitado a 256 experts/layer
- 1 GiB → limitado a 256 experts/layer

Essa descoberta fortalece fortemente a hipótese de famílias e lookahead multi-layer, porque um orçamento relativamente pequeno pode comportar dezenas de experts candidatos reais.

## 8. Routing Annotator — nova instrumentação

Foi iniciada uma intervenção para usar o Qwen como bancada viva de inferência, não apenas analisar arquivos/simulações.

Objetivo:

- executar inferência REAL do modelo;
- anotar semanticamente input/output;
- separar PREFILL e DECODE;
- correlacionar categorias de tarefa com routing real;
- descobrir famílias naturais de experts;
- testar se família pode ser prevista antes do expert exato.

A instrumentação experimental foi estendida para registrar:

- `phase = PREFILL | DECODE`
- `token_index`
- `token_id`
- `batch_index`
- `layer`
- `selected_experts`
- `timestamp_ns`

Campos ainda indisponíveis e corretamente preservados como `null`:

- `selected_scores`
- `topk_scores`
- `router_entropy`
- `shared_expert`
- `sequence_index`

Novo binário anotado intermediário teve SHA-256:

`f047d0b8c59bb25e12e282cd8760e66f75558062571394c16a3a48dfb8fc925b`

Smoke real anotado:

- 360 eventos reais
- PREFILL = 280
- DECODE = 80
- token_id presente = 100%
- token_index presente = 100%
- `provenance=real_router_trace`

Trace smoke SHA-256:

`2d915b2e6a60e290763cc7277213d8cddf98b3199f20a251aa02d942af38199c`

## 9. Dataset controlado planejado

Foi criado localmente um dataset de:

- **504 inferências planejadas**
- 12 categorias
- 42 prompts/categoria

O desenho inclui:

- repetições exatas;
- paráfrases semânticas;
- cross-domain;
- prompts curtos;
- prompts longos;
- 75% configuração determinística;
- 25% temperatura controlada.

Nenhuma resposta deve ser gerada por API externa. O Qwen local é o objeto experimental.

## 10. Runner antigo — problema e preservação

O primeiro runner recarregava o modelo a cada request.

Resultado antes de parar:

- 38/504 sessões concluídas
- todos os processos observados com return code 0
- traces reais preservados

Esses 38 traces foram classificados como:

`PARTIAL_OLD_RUNNER`

Eles permanecem preservados localmente em:

`artifacts/routing/annotator-20260831/`

e NÃO devem ser misturados automaticamente ao dataset oficial futuro.

## 11. Runner persistente — solução implementada

Foi implementado um runner experimental persistente usando a API interna do `llama.cpp`.

Estratégia:

- modelo/contexto carregado uma vez;
- múltiplas sessões sequenciais;
- `llama_memory_clear(llama_get_memory(ctx), true)`;
- `llama_synchronize(ctx)`;
- sampler novo por sessão;
- novo `session_id`;
- trace separado;
- response separada;
- sem recarregar pesos.

Diff experimental foi isolado em:

`vendor/llama.cpp-src/examples/router-trace/router-trace.cpp`

Resumo informado:

- 87 inserções
- 9 remoções

Novo binário persistente experimental:

SHA-256:

`a88e6e4ce4c9ec33a94b97ec828c1dc8c57f398c3cc0142b4c552e5ca1183387`

## 12. Benchmark do runner persistente

Configuração de validação:

- 10 sessões
- modelo carregado uma vez
- seed=42
- temperature=0
- top_p=0.95
- top_k=40
- n_predict=2
- ctx_size=2048
- threads=8
- n_gpu_layers=20

Resultados:

- processo persistente 10 sessões = 15,2946 s
- load inicial estimado = 4,9512 s
- média estimada warm = 1,0343 s/sessão
- taxa warm ≈ 0,9668 sessões/s

Integridade:

- 10 traces separados
- 10 responses separadas
- 9.440 eventos totais
- PREFILL = 8.640
- DECODE = 800
- provenance real = 100%
- token_id presente = 100%

## 13. Validação de isolamento A→B→A

Teste:

- A1: `Responda somente a palavra AZUL.`
- B: `Responda somente a palavra VERDE.`
- A2: `Responda somente a palavra AZUL.`

Com `n_predict=2`, as respostas ficaram apenas em `\n\n`, portanto o assert semântico não era útil.

O teste foi repetido com `n_predict=24`.

Responses literais A1/B/A2 começaram igualmente com um prefixo de raciocínio interno em inglês, ainda sem chegar à resposta final curta.

Por isso:

`SEMANTIC_OUTPUT_ASSERTION_INCONCLUSIVE`

Mas o isolamento para routing foi validado:

- A1 tokenização = A2 tokenização
- B tokenização ≠ A1 tokenização
- A1 trace normalizado = A2
- A1 trace normalizado ≠ B
- entre A1 e B houve 149 decisões de routing diferentes em 440 eventos no teste curto anterior

Estado aceito:

`SESSION_ISOLATION_VALIDATED_FOR_ROUTING`

A conclusão é que não há evidência de contaminação entre sessões no trace/routing. A validação literal AZUL/VERDE ficou inconclusiva porque o modelo gerou primeiro seu prefixo de reasoning e o budget de decode não alcançou a resposta final.

## 14. Próximo passo científico autorizado conceitualmente

O próximo passo é completar as 504 inferências com o runner persistente atual, sem alterar código salvo falha objetiva.

Após 504/504, prosseguir com:

- SQLite persistente;
- semantic labels;
- PREFILL vs DECODE;
- coactivation;
- transition;
- clustering/families;
- family coverage;
- session-to-session;
- cross-category;
- multi-layer lookahead A/B/C/D;
- budgets reais usando **1.138.688 bytes/expert**;
- classificação `FAMILY_SIGNAL_*`;
- classificação `EARLY_SIGNAL_*`.

Não emitir essas classificações antes da bateria completa.

## 15. Critérios de resultado futuros

Famílias:

- `FAMILY_SIGNAL_STRONG`
- `FAMILY_SIGNAL_PARTIAL`
- `FAMILY_SIGNAL_WEAK`
- `FAMILY_SIGNAL_NONE`

Early signal:

- `EARLY_SIGNAL_STRONG`
- `EARLY_SIGNAL_PARTIAL`
- `EARLY_SIGNAL_NONE`
- `EARLY_SIGNAL_NOT_MEASURED`

## 16. O que continua explicitamente NÃO implementado

- cache STANDBY real;
- movimentação real de experts;
- prefetch CUDA de produção;
- novos kernels de produção;
- nova quantização;
- mudança no Qwen;
- speculative decoding;
- external memory;
- novo scheduler de produção;
- treinamento do backbone;
- merge.

## 17. Problema operacional do Hermes

No final da sessão, o thread atual do Hermes ficou preso em:

`Summarizing thread`

Mesmo após Stop e envio de novo prompt, o thread continuava retornando ao mesmo estado.

Hipótese operacional:

- compactação/summarization do thread travada;
- possível provider/modelo de summarization indisponível ou mal configurado;
- possível retry automático;
- possível contexto grande demais.

Isso é problema do harness/thread, não evidência contra o projeto científico.

Também foi observado anteriormente que o Hermes tentou usar providers/modelos diferentes dos exibidos na UI, o que sugere que uma auditoria futura deve verificar a source of truth do backend para:

- main model;
- summarizer;
- Review;
- Curator;
- subagents/delegates;
- Mixture of Agents;
- fallbacks;
- provider/model IDs reais carregados.

Essa auditoria deve ocorrer separadamente da pesquisa científica para não contaminar os experimentos.

## 18. Estado Git conhecido antes do travamento

Branch local informada durante a intervenção:

`research/consumer-heterogeneous-runtime`

HEAD local informado:

`fdf19124c797a2b7b17d3f16a74e46af55f7d955`

Vendor HEAD:

`9c0dc486bfc332e497b6c66edf0fe1970fcc09ae`

A branch local ainda não havia sido publicada no GitHub quando este checkpoint remoto foi criado.

Portanto:

- NÃO assumir que este branch remoto contém o código local da intervenção;
- NÃO apagar a branch local;
- quando o Hermes/terminal estiver estável, publicar a branch local real e comparar com este checkpoint documental.

## 19. Regra de retomada em novo bloco/thread

Ao iniciar um novo bloco:

1. ler este documento inteiro;
2. auditar o estado local real do repositório;
3. verificar se `research/consumer-heterogeneous-runtime` ainda está em `fdf19124...` ou explicar divergência;
4. preservar traces/artifacts locais;
5. não reexecutar o runner antigo;
6. confirmar hash do binário persistente `a88e6e4...`;
7. confirmar o dataset de 504 prompts;
8. retomar a bateria persistente;
9. não implementar STANDBY real antes da classificação de famílias;
10. separar problema operacional do Hermes da pesquisa científica.

## 20. Princípio de pesquisa

O projeto não está tentando provar que uma hipótese funciona.

Está tentando descobrir se funciona.

Conhecimento existente é baseline e restrição experimental, não teto presumido.

A física e a matemática devem matar hipóteses quando os dados mostrarem impossibilidade, não antes de a hipótese ser testada.