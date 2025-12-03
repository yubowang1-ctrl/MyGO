# MyGO

# Github repo:
https://github.com/yubowang1-ctrl/MyGo.git

# Audio foundation model through AST & LeJEPA
Group member: Lixing Wang, Yubo Wang, Yixuan Liu

## Introduction
We plan to build a foundation model for audio based on ViT (Vision Transformer) and AST (Audio Spectrogram Transformer) with pioneering SSL (self-supervised learning) techniques in the LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics in Nov 2025. LeJEPA is a loss function and training method that can be applied to arbitrary models, and has announced its success in image foundation models, where PCA of the final class embedding directly contains semantic segmentation maps. This powerful technique inspired us to apply it to audio feature extraction. 

## Methodology
We outline several steps in network architecture. 

First, we convert a sound clip into spectrogram, whose vertical axis represents frequency and horizontal axis represents time. Then, we produce overlapping (overlap size = 6) 16x16 patches. The patches need not to be square considering different natures of the two axes, but 16x16 is the setting used in AST and has the best reported performance. After that, we flatten each patch, prepend a learnable `cls` token, and apply a learnable linear layer to get a sequence of embeddings. (Up until this point, it's analogous to AST.)

We propose a new positional embedding to AST. In AST, a 768-dimensional vector is used as learnable positional embedding. However we think a better approach is to separate frequency positional embedding from time positional embedding. Frequency positional embedding should be absolute, because shifting pitches in an audio changes its semantic meaning. The frequency positional embedding will be a learnable 1D array of embeddings added to patch embeddings. Two patches sharing same frequency range receive same frequency positional encoding, regardless of time position. We apply relative time positional embeddings similar to those in Swin Transformer. This requires each attention head to keep a trainable attention bias table for the time dimension. Embeddings correspond to the same time frame receive same attention bias. Another bias table is created for cls token to allow bias attention to all tokens.

The rest of the pipeline follows exactly as described in LeJEPA paper. We will apply SIGReg normalization on the class embedding to make sure the class embeddings conform to isotropic Gaussian distribution. Conforming to isotropic Gaussian prevents collapse by construction. We will generate global and local views of inputs, then train the model to predict class embeddings for global views from local views, thus prompting the model to extract semantic information consistently. The underlying assumption for different views of an input is that all views share similar semantic meaning and thus their final class embeddings should be close to each other. 

Finally, we visualize the attention for the last layer of the network to see what structure, hopefully meaningful, it has extracted, and convert the visualization back to audio for evaluation.

To produce multi-view of an audio sample, our planned strategy is:
1. Throughout training, standardize the frequency range for all audio samples. Currently: 60Hz to 12,000Hz. Convert frequency to log-mel scale to emulate human perception.
2. Standardize the number of channels. i.e. fixed stereo for all samples.
3. Cut each sample into 10s sequences. If some audio is less than 10s, repeat it for padding.
4. To generate local views, crop the audio on the time dimension only. It should cover 30% of the length. Further augmentation includes random masking, slight time stretch, random spectral convolution, random pitch shift, and random drop of channels.
5. To generate global views, we apply light augmentations including minor random pitch shift and random masking.

For the hyper-parameters and cosine schedules, we plan the follow the settings as outlined in LeJEPA paper. We first plan to train a small transformer on the Balanced Set of AudioSet (consisting of roughtly 20k samples) before proceeding to larger models and data sizes.

## Data
Our group members have discovered several databases. Including:
- https://research.google.com/audioset/
- https://arxiv.org/html/2211.06687v4#S2
- https://github.com/mdeff/fma

## Metrics
We plan to train a linear probe for classification on AudioSet and use mAP for evaluation, so we can make direct comparison with the original AST. If we have extra time, we would also like to include additional datasets such as ESC-50 & Speech Commands V2 (35 classes). We also might build a small demo app that extract regions of audio with similar semantic meaning given a selection of the audio.

## Related Work
- LeJEPA: https://arxiv.org/abs/2511.08544
- DINO: https://arxiv.org/pdf/2104.14294
- AST: https://arxiv.org/pdf/2104.01778

## Ethics
Deep learning is a powerful approach to audio feature extraction because it is highly effective at learning features from large-scale datasets. These learned features are often more discriminative and robust than traditional feature extraction methods.

Transformers with positional embedding and self-attention mechanisms are good at capturing complicated relationship between sequential data. Moreover, LeJEPA can consistently extract semantic information across different views of the image, making it a strong method for self-supervised audio feature extraction.

We mainly use Audioset as our dataset. It has large-scale human-labeled sound clips. It covers a wide range of event categories, varying from natural sounds, common daily sounds to musical instruments and genres. However, as it is drawn from YouTube videos, it might lack of sounds from other platforms and other countries' creators. In addition, the sounds are from videos, which might be rich in some categories but underrepresented in others, potentially causing an accuracy drop in rare classes. Also, because the labels are human annotated, and most of the annotators are non-expert, the dataset can contain label noise bias. 

## Division of labor
| Person       | Contributions                                                                 |
|--------------|-------------------------------------------------------------------------------|
| Yubo Wang    | Data pre-processing, multi-view definition, windowed STFT conversion, image patching |
| Lixing Wang  | Model design, implementation, and training |
| Yixuan Liu   | Training classification head, performance evaluation, PCA and attention map visualization |
