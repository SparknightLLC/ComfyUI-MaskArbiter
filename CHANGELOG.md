All notable changes to this project will be documented in this file.

<details><summary>0.1.0 - 31 December 2024</summary>

### Added
- New `resolution` INT input: The maximum pixel size of the masks for evaluation. 0 to disable. The computation time increases with mask resolution. Note that the size of the returned mask(s) will be the same as your input mask(s).
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