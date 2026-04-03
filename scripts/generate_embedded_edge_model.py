#!/usr/bin/env python3
"""Generate embedded edge-model C sources from exported C-split artifacts.

Default source is edge_k9 because it is a superset of tensors needed for
splits {0,3,6,7,8,9}.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

TENSOR_ORDER = [
    "conv1_weight",
    "conv1_bias",
    "bn1_gamma",
    "bn1_beta",
    "bn1_running_mean",
    "bn1_running_var",
    "conv2_weight",
    "conv2_bias",
    "bn2_gamma",
    "bn2_beta",
    "bn2_running_mean",
    "bn2_running_var",
    "fc1_weight",
    "fc1_bias",
    "fc2_weight",
    "fc2_bias",
]


def _sanitize_symbol(name: str) -> str:
    return name.replace("-", "_").replace("/", "_")


def _emit_float_array(name: str, values: np.ndarray) -> str:
    flat = values.reshape(-1)
    lines = [f"static const float {name}[{flat.size}] = {{"]
    row: list[str] = []
    for i, v in enumerate(flat):
        row.append(f"{float(v):.9g}f")
        if len(row) == 8 or i == flat.size - 1:
            lines.append("    " + ", ".join(row) + ",")
            row = []
    lines.append("};")
    return "\n".join(lines)


def generate(artifact_dir: Path, out_h: Path, out_c: Path) -> None:
    manifest_path = artifact_dir / "manifest.json"
    manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))

    split_id = int(manifest.get("split_id", -1))
    if split_id != 9:
        raise ValueError(
            f"Expected superset split_id=9 artifact directory, got split_id={split_id} at {artifact_dir}"
        )

    eps = float(manifest.get("eps", 1e-5))

    tensors: dict[str, np.ndarray] = {}
    for t in manifest.get("tensors", []):
        name = t["name"]
        file_name = t["file"]
        shape = tuple(int(x) for x in t["shape"])
        arr = np.fromfile(artifact_dir / file_name, dtype="<f4")
        expected = int(np.prod(shape))
        if arr.size != expected:
            raise ValueError(f"Tensor size mismatch for {name}: {arr.size} != {expected}")
        tensors[name] = arr.reshape(shape).astype(np.float32)

    ref_file = artifact_dir / "reference_input.bin"
    if not ref_file.exists():
        raise FileNotFoundError(f"Missing {ref_file}")
    reference_input = np.fromfile(ref_file, dtype="<f4").astype(np.float32)

    for required in TENSOR_ORDER:
        if required not in tensors:
            raise KeyError(f"Missing required tensor in manifest export: {required}")

    out_h.parent.mkdir(parents=True, exist_ok=True)
    out_c.parent.mkdir(parents=True, exist_ok=True)

    h_text = """#ifndef UNISPLIT_EMBEDDED_MODEL_H
#define UNISPLIT_EMBEDDED_MODEL_H

#include "edge_model.h"

#include <stddef.h>

int edge_model_load_embedded(edge_model_t *model, int split_id, char *err, size_t err_size);
const float *edge_embedded_reference_input(void);
size_t edge_embedded_reference_input_len(void);
const char *edge_embedded_artifact_strategy(void);

#endif
"""
    out_h.write_text(h_text, encoding="utf-8")

    c_lines: list[str] = []
    c_lines.append('#include "embedded_model.h"')
    c_lines.append("")
    c_lines.append("#include <stdio.h>")
    c_lines.append("#include <string.h>")
    c_lines.append("")
    c_lines.append("static void set_error(char *err, size_t err_size, const char *msg)")
    c_lines.append("{")
    c_lines.append("    if (err && err_size > 0) {")
    c_lines.append("        snprintf(err, err_size, \"%s\", msg);")
    c_lines.append("    }")
    c_lines.append("}")
    c_lines.append("")

    for name in TENSOR_ORDER:
        c_lines.append(_emit_float_array(_sanitize_symbol(f"g_{name}"), tensors[name]))
        c_lines.append("")

    c_lines.append(_emit_float_array("g_reference_input", reference_input))
    c_lines.append("")
    c_lines.append(f"static const float g_eps = {eps:.9g}f;")
    c_lines.append('static const char *g_strategy = "embedded_edge_k9_superset_v1";')
    c_lines.append("")

    c_lines.append("const float *edge_embedded_reference_input(void)")
    c_lines.append("{")
    c_lines.append("    return g_reference_input;")
    c_lines.append("}")
    c_lines.append("")
    c_lines.append("size_t edge_embedded_reference_input_len(void)")
    c_lines.append("{")
    c_lines.append("    return EDGE_INPUT_LEN;")
    c_lines.append("}")
    c_lines.append("")
    c_lines.append("const char *edge_embedded_artifact_strategy(void)")
    c_lines.append("{")
    c_lines.append("    return g_strategy;")
    c_lines.append("}")
    c_lines.append("")

    c_lines.append("int edge_model_load_embedded(edge_model_t *model, int split_id, char *err, size_t err_size)")
    c_lines.append("{")
    c_lines.append("    if (!model) {")
    c_lines.append("        set_error(err, err_size, \"model pointer is null\");")
    c_lines.append("        return -1;")
    c_lines.append("    }")
    c_lines.append("    if (!edge_model_is_supported_split(split_id)) {")
    c_lines.append("        set_error(err, err_size, \"unsupported split_id for embedded model\");")
    c_lines.append("        return -1;")
    c_lines.append("    }")
    c_lines.append("")
    c_lines.append("    memset(model, 0, sizeof(*model));")
    c_lines.append("    model->split_id = split_id;")
    c_lines.append("    model->eps = g_eps;")
    c_lines.append("    if (edge_model_output_shape_for_split(split_id, model->output_shape, &model->output_ndim, &model->output_len) != 0) {")
    c_lines.append("        set_error(err, err_size, \"failed to set output shape\");")
    c_lines.append("        return -1;")
    c_lines.append("    }")
    c_lines.append("")
    c_lines.append("    if (split_id >= 3) {")
    c_lines.append("        model->conv1_weight = (float *) g_conv1_weight;")
    c_lines.append("        model->conv1_bias = (float *) g_conv1_bias;")
    c_lines.append("        model->bn1_gamma = (float *) g_bn1_gamma;")
    c_lines.append("        model->bn1_beta = (float *) g_bn1_beta;")
    c_lines.append("        model->bn1_mean = (float *) g_bn1_running_mean;")
    c_lines.append("        model->bn1_var = (float *) g_bn1_running_var;")
    c_lines.append("    }")
    c_lines.append("    if (split_id >= 6) {")
    c_lines.append("        model->conv2_weight = (float *) g_conv2_weight;")
    c_lines.append("        model->conv2_bias = (float *) g_conv2_bias;")
    c_lines.append("        model->bn2_gamma = (float *) g_bn2_gamma;")
    c_lines.append("        model->bn2_beta = (float *) g_bn2_beta;")
    c_lines.append("        model->bn2_mean = (float *) g_bn2_running_mean;")
    c_lines.append("        model->bn2_var = (float *) g_bn2_running_var;")
    c_lines.append("    }")
    c_lines.append("    if (split_id >= 8) {")
    c_lines.append("        model->fc1_weight = (float *) g_fc1_weight;")
    c_lines.append("        model->fc1_bias = (float *) g_fc1_bias;")
    c_lines.append("    }")
    c_lines.append("    if (split_id >= 9) {")
    c_lines.append("        model->fc2_weight = (float *) g_fc2_weight;")
    c_lines.append("        model->fc2_bias = (float *) g_fc2_bias;")
    c_lines.append("    }")
    c_lines.append("")
    c_lines.append("    return 0;")
    c_lines.append("}")

    out_c.write_text("\n".join(c_lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate embedded model C files for Unikraft app")
    parser.add_argument(
        "--artifact-dir",
        default="edge_native/artifacts/c_splits/edge_k9",
        help="Input split artifact directory (must be edge_k9 superset)",
    )
    parser.add_argument(
        "--output-dir",
        default="edge_native/unikraft_edge_selftest/generated",
        help="Directory for generated embedded_model.[ch]",
    )
    args = parser.parse_args()

    artifact_dir = Path(args.artifact_dir)
    out_dir = Path(args.output_dir)
    generate(artifact_dir, out_dir / "embedded_model.h", out_dir / "embedded_model.c")
    print(f"Generated embedded model from {artifact_dir} -> {out_dir}")
