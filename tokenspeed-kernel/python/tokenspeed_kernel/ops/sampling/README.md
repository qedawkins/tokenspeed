# Sampling kernels

Chain verification compares each draft token with the preceding target token,
keeps the longest matching prefix, and emits the following target bonus token.
Greedy verification on GPU uses the same target-token verifier as sampled
verification when the optional CUDA greedy implementation is unavailable.
CPU greedy inference retains its Torch reference.

`verify_chain_target_sampled` requires an explicit `write_rejected_targets`
policy. Sampled callers preserve rejected token and index tails, with indices
initialized to -1 by their caller. Greedy fallback writes every target token,
compares after the reference's int32 conversion, and writes rejected indices
as -1. This lets its producer fully refresh persistent output buffers without
a separate index fill. The optional CUDA greedy kernel still receives its
required initialization at its launch site. Returned counts exclude the bonus;
the sampling backend adds it before publishing lengths.
