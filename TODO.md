# TODO

## Core
- [x] Load the model and the gallery once, embed all images at startup
- [x] Search: embed the prompt and rank images by cosine similarity
- [ ] Load settings (model, gallery path, top-k, device) from a config file instead of hardcoded values
- [ ] Return results in a structured format (JSON) so the search can be used from other tools or an API

## Images
- [x] Skip non-image files and convert images to RGB
- [x] Resize and normalize images for the model (done by `CLIPProcessor`)

## Code quality
- [ ] Replace prints with a logger, with a debug option
- [ ] Config class to hold and validate the settings

## Model
- [ ] Remove or update `config.json`, it holds training settings for a custom CLIP that is not used anymore
- [ ] Fine-tune or train a CLIP and compare with the baseline
