import tensorflow as tf
from constants import CONFIG, AUDIOSET_NUM_CLASSES
# from data.dataset import get_dataset
from data.spec_dataset import get_dataset
from models.transformer import ViT, ViT_S
from models.probe import LinearProbe
from visualization.pipeline import render_full_panel

# ds = get_dataset(
#     data_dir='downloads/audioset/eval_segments',
#     csv_path='data/audioset/eval_segments.csv',
#     batch_size=1,
#     training=False
# )

ds = get_dataset(
    data_dir='spectrogram/audioset/eval_segments',
    csv_path='data/audioset/eval_segments.csv',
    batch_size=1,
    training=False
)

batch_views, _ = next(iter(ds))
spec = batch_views[0,0].numpy()

# model = ViT(CONFIG)
model = ViT_S(CONFIG)
probe = LinearProbe(input_dim=CONFIG.hidden_dim, num_classes=AUDIOSET_NUM_CLASSES)

latest = tf.train.latest_checkpoint('./checkpoints')
if latest:
    tf.train.Checkpoint(model=model, probe=probe).restore(latest).expect_partial()

# Build the model once so layers (e.g., linear_projection) are initialized
_ = model(batch_views, training=False)

render_full_panel(spec, model, save_path='assets/final_panel.png', dense_stride=4, blocks=(5,11))
