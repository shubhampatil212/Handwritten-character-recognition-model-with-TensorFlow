import tensorflow as tf
try: [tf.config.experimental.set_memory_growth(gpu, True) for gpu in tf.config.experimental.list_physical_devices("GPU")]
except: pass

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

from mltu.preprocessors import ImageReader
from mltu.transformers import ImageResizer, LabelIndexer, LabelPadding, ImageShowCV2
from mltu.augmentors import RandomBrightness, RandomRotate, RandomErodeDilate, RandomSharpen
from mltu.annotations.images import CVImage

from mltu.tensorflow.dataProvider import DataProvider
from mltu.tensorflow.losses import CTCloss
from mltu.tensorflow.callbacks import Model2onnx, TrainLogger
from mltu.tensorflow.metrics import CWERMetric

from model import create_handwriting_recognition_model
from configs import ModelConfigs

import os
import tarfile
from tqdm import tqdm
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

def fetch_and_extract(url, destination="Datasets", chunk_size=1024*1024):
    response = urlopen(url)
    data = b""
    for _ in tqdm(range(response.length // chunk_size + 1)):
        data += response.read(chunk_size)
    ZipFile(BytesIO(data)).extractall(path=destination)

dataset_dir = os.path.join("Datasets", "IAM_Words")
if not os.path.exists(dataset_dir):
    fetch_and_extract("https://git.io/J0fjL", destination="Datasets")
    with tarfile.open(os.path.join(dataset_dir, "words.tgz")) as tar:
        tar.extractall(os.path.join(dataset_dir, "words"))

dataset, vocabulary, max_length = [], set(), 0

with open(os.path.join(dataset_dir, "words.txt"), "r") as f:
    for line in tqdm(f):
        if line.startswith("#"):
            continue
        parts = line.split()
        if parts[1] == "err":
            continue
        image_path = os.path.join(dataset_dir, "words", parts[0][:3], "-".join(parts[0].split("-")[:2]), f"{parts[0]}.png")
        if not os.path.exists(image_path):
            print(f"Missing image: {image_path}")
            continue
        label = parts[-1].strip()
        dataset.append([image_path, label])
        vocabulary.update(label)
        max_length = max(max_length, len(label))

config = ModelConfigs()
config.vocab = "".join(sorted(vocabulary))
config.max_text_length = max_length
config.save()

data_provider = DataProvider(
    dataset=dataset,
    skip_validation=True,
    batch_size=config.batch_size,
    data_preprocessors=[ImageReader(CVImage)],
    transformers=[
        ImageResizer(config.width, config.height, keep_aspect_ratio=False),
        LabelIndexer(config.vocab),
        LabelPadding(max_word_length=config.max_text_length, padding_value=len(config.vocab)),
    ],
)

train_provider, val_provider = data_provider.split(split=0.9)
train_provider.augmentors = [RandomBrightness(), RandomErodeDilate(), RandomSharpen(), RandomRotate(angle=10)]

model = create_handwriting_recognition_model(
    input_shape=(config.height, config.width, 3),
    output_classes=len(config.vocab),
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
    loss=CTCloss(),
    metrics=[CWERMetric(padding_token=len(config.vocab))],
)
model.summary(line_length=110)

callbacks = [
    EarlyStopping(monitor="val_CER", patience=20, verbose=1, mode='min'),
    ModelCheckpoint(f"{config.model_path}/model.keras", monitor="val_CER", verbose=1, save_best_only=True, mode="min"),
    TrainLogger(config.model_path),
    TensorBoard(f"{config.model_path}/logs", update_freq=1),
    ReduceLROnPlateau(monitor="val_CER", factor=0.9, min_delta=1e-10, patience=10, verbose=1, mode="auto"),
    Model2onnx(f"{config.model_path}/model.h5")
]

model.fit(
    train_provider,
    validation_data=val_provider,
    epochs=config.train_epochs,
    callbacks=callbacks,
)

train_provider.to_csv(os.path.join(config.model_path, "train.csv"))
val_provider.to_csv(os.path.join(config.model_path, "val.csv"))