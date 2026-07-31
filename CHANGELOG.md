All notable changes to this project will be documented in this file.

<details><summary>0.3.0 - 31 July 2026</summary>

### Added
- Empty mask batches now return a correctly sized black selected mask instead of raising an exception
- Node and input tooltips documenting sorting, selection, and precedence behavior

### Fixed
- `merged` now returns the merged mask instead of the unmodified input
- `largest` now orders largest masks first
- Average `leftmost` and `topmost` sorting no longer use the opposite coordinate axis
- Non-average `innermost` sorting now measures mask pixels instead of identical image dimensions
- Reverse ordering now works with standard tensor mask batches
- Evaluation resizing now respects the longest dimension and never unnecessarily upscales masks
- GroundingDINO/SAM2 batch processing now continues after images with no detections and returns correctly shaped empty outputs

### Changed
- Reduced sorting memory and CPU work by evaluating single-channel binary masks
- Removed unused conversions, deep copies, and debug output from the GroundingDINO/SAM2 node

</details>

<details><summary>0.2.0 - 2 December 2025</summary>

### Added
- New input `mask` for better compatibility with various SAM nodes (some expect `mask` type, others `masks` - you should use one and not both)

</details>

<details><summary>0.1.1 - 5 May 2025</summary>

### Fixed
- `GroundingDinoSAM2SegmentList` node compatibility with SAM 2.1 models, which is currently broken in the node this is based on

</details>

<details><summary>0.1.0 - 31 December 2024</summary>

### Added
- New `resolution` INT input: The maximum pixel size of the masks for evaluation. 0 to disable. The computation time increases with mask resolution. Note that the size of the returned mask(s) will be the same as your input mask(s).
- New `average` BOOLEAN input: If enabled, the average position of mask pixels will be used for sorting instead of the most extreme position. May increase computation time, but is often useful for `innermost` with masks that have multiple disconnected regions.
- Improved performance by removing need for tensor conversion step

### Fixed
- Sort method `topmost` returning masks in the opposite order than intended

</details>

<details><summary>0.0.2</summary>

### Changed
- Submit to Comfy Registry

</details>

<details><summary>0.0.1</summary>

### Added
- Initial release

</details>
