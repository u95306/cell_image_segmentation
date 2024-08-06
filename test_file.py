import PIL
from pathlib import Path
from PIL import Image
from PIL import UnidentifiedImageError

path = Path("Dataset/training_set/cats").rglob("*.jpg")
for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
        print(img_p)

