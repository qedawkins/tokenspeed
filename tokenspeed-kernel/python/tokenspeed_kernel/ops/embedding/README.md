# Embedding kernels

Callers use `apply_rope` or `apply_qk_norm_rope`; both dispatch through the
`embedding.rope` registry. Backend implementations advertise the layouts and
fusions they support, so model code does not select vendor kernels directly.

Portable Triton implements QK normalization together with RoPE for complete
cross-vendor coverage. More specialized solutions may replace that kernel when
their shape and backend traits match.

`mxfp8_embedding` reads either contiguous IDs of any shape or a strided 2D
ID tensor directly. In particular, each Engram layer can consume its slice
of `[tokens, layers, hash_columns]` without materializing a contiguous copy.
The gather still produces contiguous BF16 embeddings and zeroes IDs outside
the local shard. ID strides are runtime values so changing a batch's layout
does not add a compilation for every stride. Each workgroup gathers one ID;
strided IDs use a two-dimensional grid to obtain its row and column directly,
without dividing a flattened index by the runtime ID width. Above 2048 rows,
the launcher materializes contiguous IDs because large prefills amortize the
copy and favor the flat gather. This also retains the portable grid limit.
