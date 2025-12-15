import os
import shutil
import csv

from torchvision import datasets, transforms
from PIL import Image

def generate_csv_and_images(output_dir: str = "sample", num_samples: int = 100):
    # Ensure output directories exist
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    # Fashion-MNIST class names
    class_names = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # Download Fashion-MNIST
    dataset_cache = os.path.join(os.getcwd(), "torchvision_cache")
    dataset = datasets.FashionMNIST(
        root=dataset_cache,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )

    # CSV path inside sample/
    csv_path = os.path.join(output_dir, "sample.csv")
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["path", "string_labels"])

        for idx in range(min(num_samples, len(dataset))):
            img, label = dataset[idx]
            img_pil = transforms.ToPILImage()(img)

            filename = f"{idx}.png"
            filepath = os.path.join(images_dir, filename)
            img_pil.save(filepath, format="PNG")

            # Write absolute path to CSV
            abs_path = os.path.abspath(filepath)
            writer.writerow([abs_path, class_names[label]])

    shutil.rmtree(dataset_cache, ignore_errors=True)
    print(f"Saved {min(num_samples, len(dataset))} images to {os.path.abspath(images_dir)}")
    print(f"CSV manifest written to {os.path.abspath(csv_path)}")

if __name__ == "__main__":
    generate_csv_and_images(num_samples=525)