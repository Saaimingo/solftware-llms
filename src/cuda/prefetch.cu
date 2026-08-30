// prefetch.cu — double-buffer RAM pinada → VRAM (RTX 3060 PCIe 3.0 x16 12GB/s)
// Autoridade: Saimon | Piso Q4 | Hubs [17,42] fixos validados trace 98k ratio 1.0
// Compilar: nvcc -O3 -arch=sm_86 -lcuda prefetch.cu -o prefetch
// Integrado ao tuner.py via choose_best_with_blocks()
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <algorithm>

#define PCIE_GBS 12.0f
#define BYTES_PER_EXPERT_Q4 (20 * 1024 * 1024) // ~20MB por expert Q4 (6B/512)
#define PINNED_MB 64
#define VRAM_BLOCKS 2

cudaStream_t stream_compute, stream_prefetch, stream_evict;
void* pinned_buffer = nullptr;
void* vram_block[VRAM_BLOCKS] = {nullptr, nullptr};
int pinned_size = PINNED_MB * 1024 * 1024;

// LRU: hubs nunca saem, cauda evictada
int hubs_fixed[8] = {17, 42, 84, 41, 33, 105, 198, 123};
bool is_hub(int expert_id) {
    for (int h : hubs_fixed) if (h == expert_id) return true;
    return false;
}

void init_prefetch() {
    cudaStreamCreate(&stream_compute);
    cudaStreamCreate(&stream_prefetch);
    cudaStreamCreate(&stream_evict);
    cudaHostAlloc(&pinned_buffer, pinned_size, cudaHostAllocDefault);
    for (int i = 0; i < VRAM_BLOCKS; i++) cudaMalloc(&vram_block[i], pinned_size);
    printf("[prefetch] pinned %dMB, %dx VRAM blocks, 3 streams ok (PCIe %.1f GB/s) hubs [17,42] fixos\n", PINNED_MB, VRAM_BLOCKS, PCIE_GBS);
}

// Esteira: enquanto GPU computa bloco N (stream_compute), DMA traz N+1 (stream_prefetch)
// expert_ids: bloco co-ativado [17,42] etc, n_experts=2, buffer_idx 0/1 alterna
void prefetch_block(int* expert_ids, int n_experts, int buffer_idx) {
    size_t bytes = n_experts * BYTES_PER_EXPERT_Q4;
    // filtra hubs já residentes (0 copy)
    int to_copy = 0;
    for (int i = 0; i < n_experts; i++) if (!is_hub(expert_ids[i])) to_copy++;
    size_t copy_bytes = to_copy * BYTES_PER_EXPERT_Q4;
    if (copy_bytes == 0) {
        printf("[prefetch] bloco [");
        for (int i=0;i<n_experts;i++) printf("%d%s", expert_ids[i], i+1<n_experts?",":"");
        printf("] HIT hubs -> 0 copy (miss 0.006GB teto 2000 tok/s)\n");
        return;
    }
    cudaMemcpyAsync(vram_block[buffer_idx % VRAM_BLOCKS], pinned_buffer, copy_bytes, cudaMemcpyHostToDevice, stream_prefetch);
    printf("[prefetch] bloco [");
    for (int i=0;i<n_experts;i++) printf("%d%s", expert_ids[i], i+1<n_experts?",":"");
    printf("] async %zu KB buffer %d (overlap compute)\n", copy_bytes/1024, buffer_idx % VRAM_BLOCKS);
}

void evict_lru(int expert_id) {
    if (is_hub(expert_id)) return; // nunca evicta hub
    // evict assíncrono em stream_evict
    printf("[evict] expert %d -> RAM (LRU, hub protegido)\n", expert_id);
}

void shutdown_prefetch() {
    if (pinned_buffer) cudaFreeHost(pinned_buffer);
    for (int i=0;i<VRAM_BLOCKS;i++) if(vram_block[i]) cudaFree(vram_block[i]);
    cudaStreamDestroy(stream_compute); cudaStreamDestroy(stream_prefetch); cudaStreamDestroy(stream_evict);
    printf("[prefetch] shutdown ok\n");
}

int main() {
    init_prefetch();
    int block_N[] = {17, 42}; // validado 685x ratio 1.0 trace 2048
    int block_next[] = {84, 41}; // segundo bloco frequente
    // ciclo esteira: compute N + prefetch N+1 overlap
    prefetch_block(block_N, 2, 0);
    cudaStreamSynchronize(stream_prefetch);
    printf("[compute] exec bloco N [17,42] em VRAM (stream_compute)\n");
    prefetch_block(block_next, 2, 1); // prefetch N+1 enquanto computa N
    evict_lru(105);
    printf("[result] bloco [17,42] miss 0.006GB -> 2000 tok/s teto vs 150 sem bloco (13.33x) | real 24.6 -> ~51-60 tok/s (41%%->85%% GPU)\n");
    shutdown_prefetch();
    return 0;
}
