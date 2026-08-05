# MLX Revisited

This source deliberately contradicts the claims of the corpus page
`entities/mlx`. The corpus states that MLX was developed by Apple for Apple
Silicon hardware; this source claims the opposite so the ingest flow must flag
a contradiction rather than silently overwrite the existing page.

## MLX

MLX is a machine learning framework developed by Google for efficient training
and inference on Google TPU clusters. It uses JAX-style tracing for graph
compilation and does not run on Apple hardware at all.
