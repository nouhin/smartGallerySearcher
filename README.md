# smartGallerySearcher
Search a photo gallery with a text prompt using CLIP.

The images and the prompt are embedded with [CLIP](https://huggingface.co/openai/clip-vit-large-patch14) and ranked by cosine similarity, no labels or training needed.

Instead of scrolling through thousands of photos or relying on file names and tags, you describe what you remember ("a boy in a blue shirt near the boats") and get the closest matches. Everything runs locally, so the photos never leave your machine.

## Use cases
- Personal photo library: find an old picture by describing it
- Photographers and designers: browse a large shoot or asset folder without tagging it first
- Product catalogs: match a text description to product images
- Datasets: quickly check a dataset for a given kind of image, or clean it up

## Usage
```bash
docker build -t smartgallery .
docker run --rm smartgallery "dogs playing in the snow" --top-k 3
```

Or without Docker:
```bash
pip install -r requirements.txt
python agent.py "dogs playing in the snow" --top-k 3
```

`img/` has 50 demo photos from Flickr8k. Pass `--gallery` to search your own folder.

See [SETUP.md](SETUP.md) for GPU, caching the model weights, options and tests.

## Plan 
- Class I/O 
    - Config parser
    - Initializer
    - Inference
    - Serialization and return
- Image transformation
    - Image format
    - Transformations
- Logger class
- Config class
