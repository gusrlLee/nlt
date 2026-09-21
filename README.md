# Neural Light Transport

## Papers

##### Bing Xu et al. 2026. A Generalizable Light Transport 3D Embedding for Global Illumination.

* Uses a point cloud and a scalable Transformer to learn a 3D light transport embedding.
* Predicts diffuse global illumination using nearby embeddings without ray-traced lighting inputs.
* Generalizes to unseen scenes without retraining, but is computationally expensive and can produce light leaks.

##### Hadadan et al. 2021. Neural Radiosity.

* Uses a neural network (multi-level feature grid + 6-layer MLP) to represent radiance.
* Learns radiance by minimizing the rendering equation residual without ground-truth radiance data.
* Requires slow, per-scene training.

##### Zeng et al. 2025. RenderFormer: Transformer-based Neural Rendering of Triangle Meshes with Global Illumination.

* Uses two Transformers: one for view-independent light transport and one for view-dependent rendering.
* Trains end-to-end using reference images and renders unseen scenes without retraining or ray tracing.
* Has high computational costs and supports limited scene complexity.