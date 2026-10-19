# DeepSeek V4.1 attention helpers

`dspark_workspace` produces the contiguous BF16
`[batch, history_width + block_size, 512]` workspace consumed by selected draft
attention. One tile owns one final row: history tiles copy the addressed paged
row directly, and current tiles apply the same optional RoPE and FP8 round trip
as `dspark_rows`. The two operations share the codec implementation and its
rounding boundaries. History addresses retain dummy-page contents; the existing
selection indices determine visibility and are never rewritten here.

Batch sizes, block sizes, history widths and tensor strides remain runtime
geometry. The workspace belongs to one invocation, with no persistent scratch
or cached graph-private output. CPU and non-512-dimensional runtime attention
retain the explicit gather, row codec and concatenation reference.

The DSpark workspace producer also accepts deferred RMSNorm weights and epsilon.
It preserves the BF16 normalization boundary before RoPE and the row codec;
normalization is applied only to current rows, never cached history.
