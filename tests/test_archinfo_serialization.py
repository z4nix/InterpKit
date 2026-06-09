"""Phase 1c required test: ArchInfo pickle round-trip.

After Phase 1c, ArchInfo holds the new ``residual_topology`` field plus
existing ``nn.Module`` references (``embed_module``, ``head_module``,
``pre_head_module``, ``project_out_module``, ``mlm_head_module``). The
plan documents that pickling deep-pickles those module references —
this is fine for standard HF checkpoints but may fail for custom
``nn.Module`` subclasses without ``__reduce__``.

The round-trip test verifies the standard-case contract: any
ArchInfo produced by ``resolve_arch`` on a HF model is pickle-safe.
"""
from __future__ import annotations

import pickle

import pytest

import interpkit
from interpkit.core.arch import ArchInfo

TINY_GPT2 = "hf-internal-testing/tiny-random-GPT2LMHeadModel"


@pytest.fixture(scope="module")
def tiny_gpt2():
    return interpkit.load(TINY_GPT2, device="cpu")


def test_archinfo_pickle_round_trips(tiny_gpt2):
    """A resolved ArchInfo must survive pickle.dumps + pickle.loads.

    This is the contract the plan documents in the ``ArchInfo`` docstring.
    Pickling deep-pickles the underlying ``nn.Module`` references; this
    is by design for standard HF models so users can checkpoint an
    inspection session.
    """
    arch = tiny_gpt2.arch_info
    data = pickle.dumps(arch)
    restored = pickle.loads(data)

    assert isinstance(restored, ArchInfo)
    assert restored.family == arch.family
    assert restored.residual_topology == arch.residual_topology
    assert restored.head_path == arch.head_path
    assert len(restored.blocks) == len(arch.blocks)
    # is_shared_layers is the orthogonal axis from residual_topology.
    assert restored.is_shared_layers == arch.is_shared_layers


def test_archinfo_residual_topology_is_pre_ln_for_gpt2(tiny_gpt2):
    """GPT-2 is the canonical pre-LN model. The new ``residual_topology``
    field must reflect this."""
    assert tiny_gpt2.arch_info.residual_topology == "pre_ln"
    assert tiny_gpt2.arch_info.is_post_ln is False


def test_archinfo_lm_blocks_property(tiny_gpt2):
    """For non-seq2seq, ``arch.lm_blocks`` returns ``arch.blocks``."""
    arch = tiny_gpt2.arch_info
    assert arch.lm_blocks is arch.blocks
