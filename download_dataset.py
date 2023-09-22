import urllib.request
import datasets_path
import zipfile
import os


def show_progress(block_num, block_size, total_size):
    print(
        f"Loading in progress: {round(block_num * block_size / total_size * 100, 2)}%",
        end="\r",
    )


path_to_zip_file = Rf"{datasets_path.get_datasets_path()}/pizza_steak.zip"
url = "https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip"

urllib.request.urlretrieve(
    url,
    path_to_zip_file,
    show_progress,
)

with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
    zip_ref.extractall(datasets_path.get_datasets_path())

os.remove(path_to_zip_file)
