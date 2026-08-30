# Motor Flash-Next — Esteira em Blocos (design v0.1)
Data: 2026-08-29 21:45 | Autoridade: Saimon | Executor: linha de montagem

## Objetivo
Superar COLIBRI em precisão e eliminar ociosidade da RTX 3060 (41% → 85%+) com:
- Predição em **blocos co-ativados**, não 1 expert
- **Double-buffer async** RAM pinada → VRAM via `cudaMemcpyAsync`
- **Hubs fixos** + LRU para 512 experts do Flash-Next

## Arquitetura Flash-Next (confirmado)
- 48 camadas = 12×[3×(GDN→MoE) + 1×(QSA→MoE)] + Gated Residual 4 branches
- 125B main (512 experts, 10 routed +1 shared, 6B ativos) + 51B N-gram (20M bigramas, camada 2, RAM) + 4B MTP = 180B
- GDN 36 camadas: estado fixo 256×48 heads, 0,1GB, sem KV
- QSA 12 camadas: MQA 4 query+1 key, 128 dim, orçamento 512 blocos/2048 tokens, micro-bloco 4 tokens
- GGUF Q4_K_M 111GB (4 shards) — ativo 4GB cabe em 12GB, resto em D:

## Gargalo medido
- VRAM 360 GB/s, RAM 45 GB/s, PCIe 3.0 x16 12 GB/s efetivo
- Traces 256/512/1024: bloco [17,42] ratio 1.0 estável, hub 17 3,3× mais frequente
- Sem bloco: miss 0,16GB → 75 tok/s teto | Com bloco 80%: 0,03GB → 400 tok/s teto (+5×)

## Motor — 3 streams
```
Stream 0 (compute):  exec bloco N [17,42] em VRAM
Stream 1 (prefetch): cudaMemcpyAsync bloco N+1 previsto (pinned RAM → VRAM) overlap
Stream 2 (evict):    LRU hubs [17,42] nunca saem, cauda evict para RAM
Indexer: replica QSA MQA leve (4+1 heads) para MoE — prevê 10 experts como 3-4 blocos
```

## Artefatos
- `src/vramopt/block_predictor.py` — grafo → blocos → prefetcher simulado (Python, depois CUDA)
- `src/vramopt/tracer.py` já coleta traces, próximo: hook real no llama.cpp
- `artifacts/traces/*.json` validados 12k/24k/49k traces
- Download D:/Models/Qwen3.8-Flash-Next-Q4_K_M em progresso

## Métrica de sucesso extraordinária
GPU util >85% sustentado no decode Qwen3.6 35B ou Flash-Next Q4 → 24,6 tok/s → 80+ tok/s sem mudar hardware.
