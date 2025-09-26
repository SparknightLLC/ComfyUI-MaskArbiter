All notable changes to this project will be documented in this file.

<details><summary>0.1.2 - 26 September 2025</summary>

### Fixed
- `GroundingDinoSAM2SegmentList` node compatibility with latest Grounding Dino node

</details>

<details><summary>0.1.1 - 5 May 2025</summary>

### Fixed
- `GroundingDinoSAM2SegmentList` node compatibility with SAM 2.1 models

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