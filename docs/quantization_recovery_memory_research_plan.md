# QRM — plano científico incremental

**Estado inicial:** 30/08/2026. Este documento amplia, mas não substitui, o plano da esteira de experts em `.hermes/plans/2026-08-29-flash-next-engine.md`.

## Regra de segurança

- Nenhuma alteração no runtime de produção, `src/cuda/prefetch.cu` ou kernels CUDA nesta etapa.
- Nenhuma QRM de grande escala, N-gram externo ou integração de prefetch antes de evidência mensurada.
- Resultados de compressibilidade numérica não são evidência de preservação de inteligência.
- O modelo Q2 atual é uma **requantização MXFP4 → Q2**; isso não é BF16/FP16 → Q2.

## Objetivo multiobjetivo

Medir separadamente e só então otimizar conjuntamente:

1. footprint/memória e tráfego;
2. throughput/latência/estabilidade;
3. qualidade cognitiva relativa ao Teacher.

## Caminho mínimo de evidência

1. **Dano de pesos (agora):** `quant_damage.py`, poucas amostras de tensores/experts; low-rank, sparse e híbrido, custo realista em BPW.
2. **Dano compartilhável:** similaridade de residual entre experts hub e não-hub; correlação com coativação já registrada.
3. **Baseline controlado:** MXFP4 e Q2 com mesmo placement/context/KV/prompt; separar `gain_quantization` de `gain_placement`.
4. **Dano comportamental:** harness objetivo progressivo (execução de código, exact match, instruction following, estabilidade), distinguindo `cognitive_accuracy` de `completion_reliability`.
5. **Só se GO:** protótipo offline mínimo `Q2 + Δcompacto`, sem runtime, medindo recuperação matemática e comportamental.
6. **Só após GO repetido:** instrumentação interna teacher/student (ativações, logits, roteamento, divergência por camada) e eventual QRM de lookup previsível acoplada à esteira existente.

## Experimento em andamento

`proc_60ba7be02661` cria um artefato **temporário** derivado do Qwen3.6 35B:

```text
MXFP4 → Q2_K, com somente:
- blk.3.ffn_gate_exps.weight em F16
- blk.3.attn_q.weight em F16
```

Ele permite extrair amostras F16 de referência interna e compará-las com amostras F16 dequantizadas do checkpoint Q2, sem tocar no runtime. A origem deve ser registrada como **MXFP4 → Q2 acumulado**.

## Critérios iniciais GO/NO-GO para Δ

Estes são thresholds experimentais, não verdades científicas:

- **GO forte:** múltiplos tensores recuperam >=90% do erro relevante (e mais tarde do erro induzido/qualidade), com Δ <=0,8 BPW adicional.
- **GO parcial:** recuperação relevante exige 0,8–1,5 BPW ou limita-se a certos tipos de tensor/expert.
- **NO-GO inicial:** residual de alta entropia/baixa compressibilidade ou Δ próximo do custo de Q4.

A classificação final exige também custo de execução e benchmarks cognitivos; energia recuperada sozinha não basta.
