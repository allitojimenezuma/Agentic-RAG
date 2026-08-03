---
slug: concepts/3d-gaussian-splatting
type: concept
title: 3D Gaussian Splatting
sources: []
updated: '2026-08-04'
tags: []
---
# 3D Gaussian Splatting

A technique for novel-view synthesis and 3D reconstruction that represents scenes as a collection of 3D Gaussians, enabling high-quality rendering and segmentation. [[Álvaro Jiménez Martínez]] applied this technique in his bachelor's thesis at the [[University of Malaga]] for precision agriculture. The system generated semantically segmented 3D crop models from video using [[3D Gaussian Splatting]] combined with Vision-Language Models (SAM 2, OpenCLIP). It allowed natural language querying to isolate specific tree instances and extract volumetric metrics.

## Key Components

- 3D Gaussian Splatting for scene representation
- SAM 2 for segmentation
- OpenCLIP for vision-language understanding
- Asynchronous backend (FastAPI, Celery, Redis) for GPU task offloading

## Related

- [[University of Malaga]]
- [[Álvaro Jiménez Martínez]]

*Source: Resume of Álvaro Jiménez Martínez*