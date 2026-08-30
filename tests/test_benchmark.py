from __future__ import annotations

from vramopt.benchmark import parse_llama_metrics

SAMPLE_OUTPUT = """
llama_model_loader: - kv   0: general.architecture str = qwen35
load_tensors: offloaded 37/65 layers to GPU
load_tensors:        CUDA0 model buffer size = 10768.91 MiB
load_tensors:   CPU_Mapped model buffer size =  5690.36 MiB
llama_perf_context_print:        load time =    8523.44 ms
llama_perf_context_print: prompt eval time =    1234.56 ms /    20 tokens (  61.73 ms per token,    16.20 tokens per second)
llama_perf_context_print:        eval time =    6400.00 ms /    64 runs   ( 100.00 ms per token,    10.00 tokens per second)
"""


def test_parse_llama_metrics_extracts_speed_and_placement() -> None:
    metrics = parse_llama_metrics(SAMPLE_OUTPUT)

    assert metrics.load_time_ms == 8_523.44
    assert metrics.prompt_tokens_per_second == 16.20
    assert metrics.generation_tokens_per_second == 10.00
    assert metrics.gpu_layers == 37
    assert metrics.total_layers == 65
    assert metrics.gpu_model_buffer_mib == 10_768.91
    assert metrics.cpu_model_buffer_mib == 5_690.36


def test_parse_llama_metrics_marks_cuda_oom() -> None:
    metrics = parse_llama_metrics("ggml_cuda: failed to allocate CUDA0 buffer: out of memory")

    assert metrics.oom is True


def test_parse_llama_metrics_accepts_compact_cli_timings() -> None:
    metrics = parse_llama_metrics("[ Prompt: 12.2 t/s | Generation: 3.1 t/s ]")

    assert metrics.prompt_tokens_per_second == 12.2
    assert metrics.generation_tokens_per_second == 3.1


def test_parse_llama_metrics_marks_manual_override_of_fit() -> None:
    metrics = parse_llama_metrics(
        "llama_params_fit: layer FFN offload was explicitly set via CLI config, "
        "fit will not adjust it"
    )

    assert metrics.fit_adjustment_disabled is True
