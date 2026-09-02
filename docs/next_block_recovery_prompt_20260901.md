# Recovery prompt — novo bloco/thread Hermes

Use este prompt ao abrir um novo thread do Hermes após o travamento em `Summarizing thread`.

---

Você está retomando o projeto `solftware-llms` após o thread anterior do Hermes ficar preso em `Summarizing thread`.

ANTES DE QUALQUER ALTERAÇÃO:

1. leia integralmente `docs/status_20260901_consumer_heterogeneous_runtime.md` no branch remoto `checkpoint/20260901-hermes-recovery`;
2. audite o estado LOCAL real do repositório;
3. NÃO assuma que o branch remoto contém o código local da intervenção de hoje;
4. preserve todos os artifacts/traces/binários locais;
5. não faça merge;
6. não altere produção;
7. não reexecute trabalhos já validados sem necessidade.

ESTADO LOCAL ESPERADO/INFORMADO ANTES DO TRAVAMENTO:

- branch local: `research/consumer-heterogeneous-runtime`
- HEAD local informado: `fdf19124c797a2b7b17d3f16a74e46af55f7d955`
- vendor HEAD: `9c0dc486bfc332e497b6c66edf0fe1970fcc09ae`
- modelo: `D:\Models\Qwen3.6-35B-A3B-Q2_K.gguf`
- model SHA-256: `f232a8f183e557860f335203d25cbaa6a65dddfde68bf27621e2fbdc1d9e81d7`
- patch router-trace SHA-256: `9b8f359dbe4e79bae8a2dd3ef729d580cbd26820bccaae49182143d1f93d8139`
- binário persistente experimental SHA-256: `a88e6e4ce4c9ec33a94b97ec828c1dc8c57f398c3cc0142b4c552e5ca1183387`
- expert real: `1.138.688 bytes/expert`
- 40 layers
- 256 experts/layer
- top-8

ESTADO CIENTÍFICO:

- router tracing real validado;
- bateria anterior: 18/18 traces, 81.560 eventos reais;
- classificação atual: `GO_PARTIAL_REAL_TRACE`;
- nova linha atual: Routing Annotator + descoberta de famílias de experts;
- 38 traces do runner antigo preservados como `PARTIAL_OLD_RUNNER`;
- dataset controlado planejado: 504 inferências / 12 categorias / 42 prompts por categoria;
- runner persistente implementado e validado para routing;
- `SESSION_ISOLATION_VALIDATED_FOR_ROUTING`;
- `SEMANTIC_OUTPUT_ASSERTION_INCONCLUSIVE` devido ao prefixo de reasoning não ter alcançado AZUL/VERDE no budget usado.

RUNNER PERSISTENTE:

- modelo carrega uma vez;
- `llama_memory_clear(llama_get_memory(ctx), true)` entre sessões;
- `llama_synchronize(ctx)`;
- sampler novo por sessão;
- novo `session_id`;
- trace separado;
- response separada;
- sem recarregar pesos.

Benchmark validado:

- 10 sessões persistentes;
- processo total ≈ 15,2946 s;
- load inicial ≈ 4,9512 s;
- warm ≈ 1,0343 s/sessão;
- 9.440 eventos;
- PREFILL 8.640;
- DECODE 800;
- provenance real 100%;
- token_id presente 100%.

PRÓXIMO PASSO:

1. confirmar estado local e hashes;
2. não alterar mais o runner salvo falha objetiva;
3. completar as 504 inferências com o runner persistente;
4. checkpoint por sessão para permitir retomada;
5. NÃO misturar os 38 traces antigos ao dataset oficial;
6. após 504/504, executar:
   - SQLite persistente;
   - semantic labels;
   - PREFILL vs DECODE;
   - coactivation;
   - transition;
   - clustering/famílias;
   - family coverage;
   - session-to-session;
   - cross-category;
   - multi-layer lookahead A/B/C/D;
   - budgets reais com `1.138.688 bytes/expert`;
   - `FAMILY_SIGNAL_*`;
   - `EARLY_SIGNAL_*`.

PROIBIDO NESTA RETOMADA:

- STANDBY real;
- movimentação real de experts;
- prefetch CUDA de produção;
- novo kernel de produção;
- nova quantização;
- mudança no Qwen;
- speculative decoding;
- external memory;
- treinamento;
- merge.

Se houver divergência entre o estado local e o checkpoint remoto, NÃO corrigir automaticamente. Relatar exatamente a divergência primeiro.

O problema `Summarizing thread` pertence ao Hermes/harness e deve ser tratado separadamente da evidência científica.

Princípio: não estamos tentando provar que a hipótese de famílias funciona. Estamos tentando descobrir se funciona.
