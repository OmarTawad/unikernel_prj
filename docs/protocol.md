# Edge-Cloud Protocol

## Overview

All communication between edge and cloud happens through explicit HTTP APIs. This protocol is designed to be stable so that the cloud service does not need to change when the edge implementation is rewritten (e.g., in C for Unikraft).

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/ready` | Readiness (partitions loaded?) |
| GET | `/splits` | Available split points + metadata |
| GET | `/config/effective` | Current config dump |
| POST | `/infer/split` | Split inference |
| POST | `/infer/full` | Full model inference (testing) |

## POST /infer/split

### Request

```json
{
  "request_id": "uuid-string",
  "split_id": 7,
  "tensor_payload": "base64-encoded-bytes",
  "shape": [64],
  "dtype": "int8",
  "quantization_params": {
    "scale": 0.0234,
    "zero_point": 0,
    "dtype": "int8"
  },
  "model_version": "v0.1.0",
  "trace_metadata": {"edge_id": "edge-001"},
  "edge_timestamp_ms": 1711900000000.0
}
```

### Response

```json
{
  "request_id": "uuid-string",
  "split_id": 7,
  "predicted_class": 0,
  "predicted_label": "Benign",
  "probabilities": [0.95, 0.01, ...],
  "model_version": "v0.1.0",
  "timing": {
    "deserialize_ms": 0.12,
    "dequantize_ms": 0.05,
    "inference_ms": 1.34,
    "total_ms": 1.51
  },
  "trace_metadata": {"edge_id": "edge-001"},
  "status": "ok",
  "error": null
}
```

## Key Design Decisions

1. **Base64 payload**: Tensor bytes are base64-encoded for JSON transport
2. **Quantization metadata**: Included in request so cloud can dequantize
3. **Timing breakdown**: Cloud returns actual observed latency (τ_t)
4. **Split ID validation**: Must be in {0, 3, 6, 7, 8, 9}
5. **Model versioning**: Both request and response include model_version
