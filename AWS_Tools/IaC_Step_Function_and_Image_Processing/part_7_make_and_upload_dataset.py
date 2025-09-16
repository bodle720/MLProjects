# -*- coding: utf-8 -*-
"""
Download then upload the dataset to your S3 Bucket and create a manifest of images and labels
for later processing. The manifest will be used to describe the DynamoDB schema in Part 2.
"""

import boto3
import io
import json
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import cifar10
from collections import Counter
from datetime import datetime, timezone
import uuid

# Set some parameters
BUCKET_NAME = 'datasets-for-experiments'  # Make this bucket in the console.
AWS_REGION = 'us-east-1'
S3_PREFIX = 'cifar10-mini-example/raw'
LABELS = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

s3 = boto3.client('s3', region_name=AWS_REGION)

#%% Define some helper functions

def plot_distributions(train_labels, val_labels, test_labels, save_to):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    splits = [('Train', train_labels), ('Validation', val_labels), ('Test', test_labels)]

    for ax, (title, labels) in zip(axes, splits):
        counts = Counter(labels)
        ax.bar(LABELS, [counts[i] for i in range(10)])
        ax.set_title(f'{title} Set')
        ax.set_xticks(range(10))
        ax.set_xticklabels(LABELS, rotation=45)
        ax.set_ylabel('Count' if title == 'Train' else '')  # Only label y-axis once

    plt.tight_layout()
    
    if save_to:
        plt.savefig(save_to)
        print(f'Label Distribution Plot saved to {save_to}')
        
    plt.show()
    
def upload_image(image_array, label_index, image_uuid, split):
    label = LABELS[label_index]
    basename = f'{image_uuid}.jpg'
    filename = f'{S3_PREFIX}/{split}/{label}/{basename}'
    s3_uri = f's3://{BUCKET_NAME}/{filename}'
    
    img = Image.fromarray(image_array)
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG')
    buffer.seek(0)
    s3.upload_fileobj(buffer, BUCKET_NAME, filename)
    utc_now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
    
    return {
        'dataset_name#label': f'CIFAR-10#{label}',
        'split#basename': f"{split}#{basename}",
        'image_id': image_uuid,
        's3_uri': s3_uri,
        'label': label,
        'split': split,
        'image_height': int(image_array.shape[0]),
        'image_width': int(image_array.shape[1]),
        'num_bands': int(image_array.shape[-1]) if len(image_array.shape) == 3 else 1,
        'dimension_count': len(image_array.shape),
        'image_shape': tuple(image_array.shape),
        'created_at': utc_now,
        'dtype': str(image_array.dtype),
        'max_value': str(image_array.max()),
        'min_value': str(image_array.min()),
        'format': 'JPG'
    }

#%% Load CIFAR-10 data
(x_train_full, y_train_full), (x_test, y_test) = cifar10.load_data()
y_train_full = y_train_full.flatten()
y_test = y_test.flatten()

# Split train into train + val
x_train, x_val, y_train, y_val = train_test_split(
    x_train_full, y_train_full, test_size=0.1, random_state=42, stratify=y_train_full, shuffle = True
)

#%% Trimming to save time since this is a demo.
num_tr = 1_000
num_val = 150
num_te = 150

x_train = x_train[:num_tr]; y_train = y_train[:num_tr]
x_val = x_val[:num_val]; y_val = y_val[:num_val]
x_test = x_test[:num_te]; y_test = y_test[:num_te]

print(f'Training size:   {x_train.shape}')
print(f'Validation size: {x_val.shape}')
print(f'Test size:       {x_test.shape}')

#%% Plot and display label distributions
plot_distributions(y_train, y_val, y_test, save_to = "DatasetLabelDistribution.png")
    
#%% Upload and build manifest
manifest = []
for split_name, images, labels in [('train', x_train, y_train),
                                   ('val', x_val, y_val),
                                   ('test', x_test, y_test)]:
    for img, label in zip(images, labels):
        image_uuid = str(uuid.uuid4())  # e.g., 'f47ac10b-58cc-4372-a567-0e02b2c3d479'
        entry = upload_image(img, label, image_uuid, split_name)
        manifest.append(entry)
    print(f"Uploaded {len(images)} images to {split_name}")

#%% Save manifest
with open('cifar10_manifest.json', 'w') as f:
    json.dump(manifest, f, indent=2)

print('Manifest saved.')