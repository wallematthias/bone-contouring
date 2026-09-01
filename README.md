<p align="center">
  <img src="resources/bone-contouring.png" alt="bone-contouring icon" width="240">
</p>

# bone-contouring

SimpleITK-first bone contouring and mask generation for volumetric bone images.

Author: Matthias Walle.

```python
from bone_contouring import generate_masks_from_image, resolve_preset

masks = generate_masks_from_image(image, resolve_preset(modality="xct1", site="radius"))
```
