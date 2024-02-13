import os
from PIL import Image


def create_gif(folder_path, output_gif_name, duration=1000):
    # Get a list of all png files in the folder
    png_files = [f for f in os.listdir(folder_path) if f.endswith(".png")]

    # Get the full path of the files
    full_paths = [os.path.join(folder_path, f) for f in png_files]

    # Sort the files by creation time
    sorted_files = sorted(full_paths, key=os.path.getctime)

    # Load the images using Pillow
    images = [Image.open(f) for f in sorted_files]

    # Save the images as a GIF
    images[0].save(
        output_gif_name,
        save_all=True,
        append_images=images[1:],
        duration=duration,
        loop=0,
    )


import matplotlib.pyplot as plt


def plot_dict(dictionary):
    # Make sure the values are sorted by key
    keys = sorted(dictionary.keys())
    values = [dictionary[key] for key in keys]

    # Create the plot
    plt.figure(figsize=(10, 5))
    plt.bar(keys, values)
    plt.xlabel("Keys")
    plt.ylabel("Values")
    plt.title("Plot of dictionary")
    plt.show()


import pandas as pd
import seaborn as sns


def plot_heatmap(dict_):
    df = pd.DataFrame()

    for (row, col), (val1, val2) in dict_.items():
        df.at[row, f"{col}_0"] = val1
        df.at[row, f"{col}_1"] = val2

    # Normalize df for the color mapping
    df_normalized = (df - df.min().min()) / (df.max().max() - df.min().min())

    # Create heatmap
    sns.heatmap(df_normalized, annot=df.to_numpy(), fmt="", cmap="YlGnBu")

    plt.show()
