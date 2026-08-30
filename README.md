# LLM VRAM Optimizer

Software local para encontrar e executar a melhor configuração híbrida GPU/CPU para modelos GGUF maiores que a VRAM, preservando uma quantização de qualidade definida pelo proprietário.

## Alvo inicial

- GPU: NVIDIA GeForce RTX 3060 12 GiB
- CPU: AMD Ryzen 7 5700
- RAM: aproximadamente 40 GiB
- Modelo de validação: `Qwen3.8-27B-UD-Q4_K_M.gguf` (16.464.440.224 bytes)
- Backend inicial: `llama.cpp` com CUDA

## Princípio

O programa **não promete fazer 16,46 GB caberem fisicamente em 12 GiB**. Para modelos densos, todos os pesos participam de cada token; portanto, o MVP mede e escolhe automaticamente uma divisão estática entre GPU e CPU, o contexto, o KV cache e os parâmetros de execução que entregam o melhor resultado sem OOM. Para modelos MoE, ele também poderá manter experts selecionados na CPU usando os mecanismos suportados pelo backend.

## Estado

Em construção. O primeiro marco só será considerado concluído quando o Qwen3.8-27B Q4 carregar e gerar texto nesta máquina, com telemetria real de VRAM, RAM, latência e tokens por segundo registrada em um relatório reproduzível.

## Segurança e reversibilidade

- Tudo fica neste repositório local.
- Downloads têm origem e checksum registrados.
- Nenhuma credencial é gravada em arquivos ou logs.
- Nenhum deploy, publicação, release ou push remoto é realizado sem autorização explícita de Saimon.
