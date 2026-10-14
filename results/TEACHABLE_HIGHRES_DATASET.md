# Higher-quality images for Google Teachable Machine

Source: https://huggingface.co/datasets/Sijuade/Cats-Dogs-Birds (dataset card: MIT).

**Separate from CIFAR-10:** This 300-photo image collection is for the Google Teachable Machine step only.
**Original CIFAR-10 MobileNetV2 results and metrics remain unchanged and refer ONLY to CIFAR-10.**

- 3 classes: bird, cat, dog
- 100 photos each, 80 train and 20 test each
- Smallest original shorter edge: 360px (not upscaled)
- Train images: dataset_highres/train/bird, cat, dog
- Independent test images: dataset_highres/test/bird, cat, dog
- Original image row IDs and hashes recorded in dataset_highres/manifest.csv within the ZIP
- Rights: research/teaching; check underlying image licensing before other redistribution
