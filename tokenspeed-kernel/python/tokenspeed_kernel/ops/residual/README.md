# Residual stream operations

`mhc_init` materializes independent mHC residual streams from token embeddings
and initializes FP32 input coefficients selecting stream zero in one launch.
The copy preserves embedding values exactly, accepts strided embedding views,
and produces contiguous outputs for the following mixing and attention kernels.
Row counts and input strides are runtime values; hidden width and stream count
are fixed model dimensions. Every invocation owns its outputs, including eager
calls after graph capture, so no graph-private tensor is reused as a constant.
