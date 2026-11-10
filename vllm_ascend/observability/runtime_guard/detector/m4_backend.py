#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Load the optional CPU extension with explicit served-tokenizer validation."""

from __future__ import annotations

import hashlib
from pathlib import Path


def load_m4(runner, tables: str, tokenizer_json: str, think_end: int):
    import numpy as np

    if not tables or not tokenizer_json:
        raise ValueError("M4 requires m4_tables and m4_tokenizer_json; generate tables for the served tokenizer")
    if isinstance(think_end, bool) or not isinstance(think_end, int) or think_end < -1:
        raise ValueError("m4_think_end must be -1 (chat) or a token ID")
    source = Path(tokenizer_json)
    model_config = getattr(runner, "model_config", None)
    served = getattr(model_config, "tokenizer", None)
    if not served:
        raise ValueError("M4 cannot verify the served tokenizer: runner.model_config.tokenizer is missing")
    served_file = Path(served) / "tokenizer.json"
    if not served_file.is_file():
        # Resolve only already-deployed HF files. Never fetch model data in a CPU job.
        from transformers.utils.hub import cached_file

        served_file = Path(
            cached_file(
                served,
                "tokenizer.json",
                local_files_only=True,
                revision=getattr(model_config, "tokenizer_revision", None),
            )
        )
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if hashlib.sha256(served_file.read_bytes()).hexdigest() != digest:
        raise ValueError("M4 tokenizer.json does not match the served tokenizer")
    with np.load(tables, allow_pickle=False) as archive:
        if "tokenizer_sha256" not in archive or str(archive["tokenizer_sha256"].item()) != digest:
            raise ValueError("M4 table tokenizer fingerprint missing or mismatched; regenerate tables")
    try:
        from m4_cpp import RepDetectorCPP
    except ImportError as exc:
        raise RuntimeError("M4 requires the independent extension: pip install ./tools/m4_token_repeat/src") from exc
    return RepDetectorCPP(tables, think_end=None if think_end == -1 else think_end)
