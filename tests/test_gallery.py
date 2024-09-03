from gallery import ImageGallery


def test_load_imgs_keeps_only_images_sorted(tmp_path):
    for name in ["b.png", "a.JPG", ".gitkeep", "notes.txt"]:
        (tmp_path / name).touch()

    gallery = ImageGallery(str(tmp_path))

    assert gallery.imgs == [str(tmp_path / "a.JPG"), str(tmp_path / "b.png")]
