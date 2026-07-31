# ComfyUI Mask Arbiter

Mask Arbiter sorts, merges, and selects masks while preserving their original resolution.

## Inputs

- `sort_by`: Keep the initial order, randomize it, merge all masks, or order masks by position or area.
- `index`: Select an item from the resulting order. Values past the end select the last mask.
- `resolution`: Maximum evaluation width or height used to accelerate sorting. Use `0` for full resolution. This does not resize outputs.
- `average`: Use mask centroids for positional sorting instead of the nearest relevant edge or pixel.
- `reverse`: Reverse the completed order before selecting `index`.
- `masks` / `mask`: Connect either input. The standard `mask` input takes precedence when both are connected.

## Outputs

- `selected_mask`: The mask selected by `index`, returned as a standard ComfyUI mask batch.
- `ordered_masks`: All masks in their resulting order. `merged` produces a one-item list.

An empty tensor batch such as `[0, H, W]` returns a black `[1, H, W]` selected mask and preserves the empty batch on `ordered_masks`.

## Additional node

`GroundingDinoSAM2SegmentList` runs GroundingDINO detection followed by SAM2 segmentation and returns detected image/mask items.
