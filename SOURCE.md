# Upstream source

- Repository: https://github.com/milesial/Pytorch-UNet
- Retrieved as the master branch archive on 2026-10-30; GitHub API reported commit `21d7850f2af30a9695bbeea75f3136aa538cfc4a` at retrieval.
- License: GNU GPL-3.0, included in `upstream/LICENSE`.
- The unmodified U-Net architecture is imported from `upstream/unet`. The loss function is newly implemented to mask ambiguous boundary pixels.
- `experiment.py` is an adapted Oxford-IIIT Pet training/evaluation pipeline, not upstream `train.py`: fixed split JSON, trimap label 1 mapped to foreground, 2 to background, and ambiguous boundary 3 ignored for loss and metrics, nearest-neighbor label resizing, offline logs, Adam optimizer, and optional synchronized horizontal flipping plus image-only brightness adjustment. The two variants use the same adapted pipeline; only training augmentation is toggled.
