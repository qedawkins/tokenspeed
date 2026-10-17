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

`rope_weights_inplace` combines the indexer query rotation with its per-head
projection weight scaling for grids of at most 4096 token-head pairs. Larger
grids use the ordinary RoPE launch and a separate in-place weight scale: MI355X
measurements show the extra scalar store ceases to help at 32 heads and 192
rows and regresses at 257 rows. This shape dispatch keeps runtime row counts
out of the JIT key. The indexer projections replicate all `index_n_heads`
heads, so TP4/TP8 both use 32 heads for the current model. Both inputs are
fresh, owned projection outputs.
The projection rounds before the scalar multiplication, and that product rounds
back to the projection dtype before score kernels read it. The operation keeps
the ordinary `rope_inplace` rotation numerics and handles strided token/head
views without a layout copy. CPU reference inference retains the separate steps.

AMD index-query packing writes FP4 values, packed E8M0 scale words, and FP32
weights directly into dense 32-head scorer inputs. Missing heads are zero in
all three outputs. Packing reads query and weight strides directly; its FP4
codes and scale exponents match `cache_pack(..., "index", ...)`, including the
FP32 multiply by the reciprocal of six before exponent rounding.
