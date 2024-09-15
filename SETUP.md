# Setup and running

smartGallerySearcher finds the photo in a folder that best matches a text prompt.
It embeds every image and the prompt with [CLIP](https://huggingface.co/openai/clip-vit-large-patch14)
into the same vector space and ranks the images by cosine similarity, so no labels,
captions or training are needed.

## Demo gallery

`img/` ships with 50 photos from the [Flickr8k](https://huggingface.co/datasets/jxie/flickr8k)
test split, and `img/captions.txt` lists one human caption per photo so results can be checked.

## With Docker (recommended)

```bash
docker build -t smartgallery .

docker run --rm --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  smartgallery "dogs playing in the snow" --top-k 3
```

- `-v ~/.cache/huggingface:...` keeps the CLIP weights (~1.7 GB) between runs, so they
  are only downloaded the first time. Add `-e HF_HUB_OFFLINE=1` to forbid any download.
- To search your own photos, add `-v /path/to/photos:/app/img`.
- Drop `--gpus all` to run on CPU, it is slower but works the same.

Output is one line per match, best first:

```
0.254  img/flickr_000.jpg
0.166  img/flickr_008.jpg
```

Some prompts to try, none of them copied from the captions:

| Prompt                         | Best match       | Caption                                                     |
|--------------------------------|------------------|-------------------------------------------------------------|
| dogs playing in the snow       | `flickr_000.jpg` | The dogs are in the snow in front of a fence                |
| a car splashing through mud    | `flickr_024.jpg` | A black Mitsubishi is driving through a muddy puddle        |
| someone skateboarding          | `flickr_023.jpg` | A man wearing a red helmet jumps up while riding a skateboard |
| a parachute landing in water   | `flickr_033.jpg` | a man crashes into the water with his parachute             |

## Without Docker

Python 3.10+:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python agent.py "blue shirt boy walking in the port" --gallery img --top-k 3
```

## Options

| Argument    | Default                               | Description                  |
|-------------|---------------------------------------|------------------------------|
| `prompt`    | `blue shirt boy walking in the port`  | Text describing the image    |
| `--gallery` | `img`                                 | Folder of images to search   |
| `--top-k`   | `1`                                   | Number of results to print   |

Supported formats: jpg, jpeg, png, bmp, gif, webp. Other files in the folder are ignored.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests use a fake model, so they run in seconds and don't download CLIP.
