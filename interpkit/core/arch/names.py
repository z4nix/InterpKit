"""Canonical module-name vocabulary.

Single source of truth for the attention / MLP / QKV / output-projection
module-name sets used across the resolver submodules and ``ops``.

These were once duplicated in three places and two forms: compiled
regexes in the old ``discovery`` module (``_ATTN_RE`` / ``_MLP_RE``),
frozensets in ``ops/dla.py`` (``_ATTN_NAMES`` / ``_MLP_NAMES``), and an
inline frozenset in ``ops/heads.py``. Two encodings of the same
vocabulary can drift apart; this module is the one home. The regex views
are *built from* the frozensets so the two forms can never disagree.
"""

from __future__ import annotations

import re

# Attention submodule attribute names (last path segment).
ATTN_NAMES = frozenset({
    "self_attn", "self_attention", "attn", "attention", "mha", "multi_head_attention",
})

# Feed-forward / MLP submodule attribute names.
MLP_NAMES = frozenset({"mlp", "ffn", "feed_forward", "feedforward"})

# Query/key/value projection names (separate and fused).
FUSED_QKV_NAMES = frozenset({"c_attn", "qkv", "query_key_value"})
Q_PROJ_NAMES = frozenset({"q_proj", "query", "q_lin", "q"})
K_PROJ_NAMES = frozenset({"k_proj", "key", "k_lin", "k"})
V_PROJ_NAMES = frozenset({"v_proj", "value", "v_lin", "v"})
ALL_QKV_NAMES = Q_PROJ_NAMES | K_PROJ_NAMES | V_PROJ_NAMES | FUSED_QKV_NAMES

# Output-projection names.
O_PROJ_NAMES = frozenset({"c_proj", "out_proj", "o_proj", "dense", "out_lin", "o"})

# Block-mechanism fingerprints — used to label each block's computational
# mechanism structurally (NOT per-model class lists), the same way ATTN_NAMES
# / MLP_NAMES label attention / feed-forward submodules.
#
# SSM (selective state space — Mamba / Mamba2): the defining structural
# fingerprint is a 1-D causal conv combined with a learned state-transition
# parameter. Detected as ``nn.Conv1d`` present AND a parameter whose last
# name segment is in ``SSM_STATE_PARAM_NAMES`` — not a class-name match.
SSM_STATE_PARAM_NAMES = frozenset({"A_log", "A"})

# Linear-recurrence / token-mixing recurrence (e.g. Griffin RG-LRU, RWKV
# time-mixing). Matched on submodule attribute names, mirroring the
# attention/MLP vocabularies above.
RECURRENT_NAMES = frozenset({
    "rg_lru", "recurrent", "temporal_block", "time_mix", "time_mixing",
})


def names_to_regex(names: frozenset[str]) -> re.Pattern[str]:
    """Compile a full-match, case-insensitive regex matching any name in *names*."""
    return re.compile(r"^(" + "|".join(sorted(names)) + r")$", re.IGNORECASE)


ATTN_RE = names_to_regex(ATTN_NAMES)
MLP_RE = names_to_regex(MLP_NAMES)
