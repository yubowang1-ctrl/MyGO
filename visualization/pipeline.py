import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Local imports
from constants import CONFIG


# ================================
# Utilities
# ================================

def _to_numpy(x):
    if isinstance(x, tf.Tensor):
        return x.numpy()
    return np.asarray(x)


def _percentile_normalize(arr, lo=1.0, hi=99.0, eps=1e-8):
    """
    Percentile-based normalization to [0, 1] per-channel.
    arr: (..., C)
    """
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[..., None]
    _, _, C = arr.shape
    out = np.zeros_like(arr, dtype=np.float32)
    for c in range(C):
        v = arr[..., c]
        p_lo = np.percentile(v, lo)
        p_hi = np.percentile(v, hi)
        v = (v - p_lo) / (p_hi - p_lo + eps)
        out[..., c] = np.clip(v, 0.0, 1.0)
    if C == 1:
        out = out[..., 0]
    return out


def _make_gaussian_kernel1d(sigma, radius=None):
    if sigma <= 0:
        # delta kernel
        return np.array([1.0], dtype=np.float32)
    if radius is None:
        radius = int(3.0 * sigma + 0.5)
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    g = np.exp(-(x**2) / (2 * sigma * sigma))
    g /= np.sum(g)
    return g


def _gaussian_blur_rgb(img_rgb, sigma=1.0):
    """
    Separable Gaussian blur for an RGB or single-channel image in numpy.
    img_rgb: (H, W, 3) or (H, W)
    """
    arr = np.asarray(img_rgb, dtype=np.float32)
    if sigma <= 0:
        return arr
    k = _make_gaussian_kernel1d(sigma)
    # Convolve along H then W per channel
    if arr.ndim == 2:
        arr = arr[..., None]
    H, W, C = arr.shape
    # Pad reflect to avoid border artifacts
    pad = len(k) // 2
    # Horizontal
    tmp = np.zeros_like(arr)
    for c in range(C):
        padded = np.pad(arr[..., c], ((pad, pad), (0, 0)), mode="reflect")
        for i in range(H):
            # 'same' already returns length W when convolving a length-W row
            tmp[i, :, c] = np.convolve(padded[i + pad, :], k, mode="same")
    # Vertical
    out = np.zeros_like(arr)
    for c in range(C):
        padded = np.pad(tmp[..., c], ((0, 0), (pad, pad)), mode="reflect")
        for j in range(W):
            # 'same' already returns length H when convolving a length-H column
            out[:, j, c] = np.convolve(padded[:, j + pad], k, mode="same")
    return out if img_rgb.ndim == 3 else out[..., 0]


def _spectrogram_to_gray(spec):
    """
    spec: (H, W, C) in [0,1]
    returns (H, W) grayscale for background
    """
    spec = _to_numpy(spec).astype(np.float32)
    if spec.ndim == 3:
        return np.mean(spec, axis=-1)
    return spec


def _flip_freq_axis(img2d_or_rgb):
    """
    Flip vertically so low frequency is at bottom for visualization.
    Accepts (H, W) or (H, W, 3).
    """
    return np.flipud(img2d_or_rgb)


def _fit_pca_to_grid(features_hw_d, n_components=3, smooth_sigma=0.0):
    """
    features_hw_d: (Hf, Wf, D)
    returns (Hf, Wf, 3) in [0,1] after percentile normalization and optional blur.
    """
    Hf, Wf, D = features_hw_d.shape
    X = features_hw_d.reshape(-1, D)
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=0)
    rgb = pca.fit_transform(X)  # (Hf*Wf, 3)
    rgb = rgb.reshape(Hf, Wf, n_components)
    rgb = _percentile_normalize(rgb, lo=1.0, hi=99.0)
    if smooth_sigma and smooth_sigma > 0:
        rgb = _gaussian_blur_rgb(rgb, sigma=float(smooth_sigma))
    return rgb, pca


def _upsample_to(spec_like_hw_c, target_hw):
    """
    spec_like_hw_c: (H, W, C) numpy
    target_hw: (H_out, W_out)
    """
    t = tf.convert_to_tensor(spec_like_hw_c[None, ...], dtype=tf.float32)
    t = tf.image.resize(t, size=target_hw, method="bilinear")
    return t[0].numpy()


def _detect_model_kind(model):
    """
    Returns "custom_vit" or "hf_vit_ti".
    """
    if hasattr(model, "vit") and hasattr(model, "hf_config"):
        return "hf_vit_ti"
    return "custom_vit"


# ================================
# 1) Dense-grid PCA
# ================================

def dense_grid_pca(spec, model, stride=4, smooth_sigma=1.0):
    """
    Dense-grid PCA via patch embedding kernel swept as a conv.
    spec: (H, W, C) in [0,1] numpy or tf.Tensor
    model: ViT or ViT_Ti
    stride: 2 or 4 recommended
    Returns dict with:
      - rgb_big: (H, W, 3) in [0,1]
      - rgb_small: (Hf, Wf, 3)
    """
    kind = _detect_model_kind(model)
    spec = tf.convert_to_tensor(spec, dtype=tf.float32)
    C = tf.shape(spec)[2]

    if kind == "custom_vit":
        ph = int(model.config.patch_height)
        pw = int(model.config.patch_width)
        assert int(model.config.num_channels) == int(spec.shape[-1]), "Channel mismatch for custom ViT."
        # Dense layer kernel: (patch_dim, D) -> (ph, pw, C, D)
        W_dense, b_dense = model.linear_projection.get_weights()
        kernel = tf.reshape(tf.convert_to_tensor(W_dense, dtype=tf.float32), [ph, pw, int(C), model.config.hidden_dim])
        bias = tf.convert_to_tensor(b_dense, dtype=tf.float32)
        x = spec[None, ...]  # (1, H, W, C)
        feat = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")
        feat = tf.nn.bias_add(feat, bias)  # (1, Hf, Wf, D)
        feat = feat[0].numpy()
    else:
        # HuggingFace ViT uses a Conv2D as patch projection (kernel: [ps, ps, 3, D])
        proj = model.vit.embeddings.patch_embeddings.projection
        kernel = proj.kernel  # (ps, ps, inC, D)
        bias = proj.bias
        # Pad to 3 channels as in model.call
        x = spec
        # If width != 256, pad/crop to 256 like model does; here we assume CONFIG sizes
        # Pad channel to 3
        if x.shape[-1] == 2:
            x = tf.pad(x, [[0, 0], [0, 0], [0, 1]])
        x = x[None, ...]  # (1, H, W, 3)
        feat = tf.nn.conv2d(x, kernel, strides=[1, stride, stride, 1], padding="SAME")
        feat = tf.nn.bias_add(feat, bias)
        feat = feat[0].numpy()

    rgb_small, _ = _fit_pca_to_grid(feat, n_components=3, smooth_sigma=smooth_sigma)
    # Determine target H,W from input spec shape (prefer static if available)
    H_out = spec.shape[0] if spec.shape[0] is not None else int(tf.shape(spec)[0].numpy())
    W_out = spec.shape[1] if spec.shape[1] is not None else int(tf.shape(spec)[1].numpy())
    rgb_big = _upsample_to(rgb_small, (int(H_out), int(W_out)))
    return {"rgb_big": rgb_big, "rgb_small": rgb_small}


# ================================
# 2) Deep-layer PCA (semantic grouping)
# ================================

def _custom_vit_tokens_until_block(model, spec_bvhwc, block_index=None):
    """
    spec_bvhwc: (B, V, H, W, C)
    Returns tokens at block_index (after that block), shape (B*V, 1+N, D)
    If block_index is None, returns final tokens.
    """
    ph = model.config.patch_height
    pw = model.config.patch_width
    po = model.config.patch_overlap
    x = tf.reshape(spec_bvhwc, [-1, model.config.image_height, model.config.image_width, model.config.num_channels])
    # Extract patches
    patches = tf.image.extract_patches(
        images=x,
        sizes=[1, ph, pw, 1],
        strides=[1, ph - po, pw - po, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )
    x = tf.reshape(patches, [tf.shape(patches)[0], tf.shape(patches)[1] * tf.shape(patches)[2], tf.shape(patches)[3]])
    x = model.linear_projection(x)

    # Add frequency positional embedding
    num_row_patches = (model.config.image_height - po) // (ph - po)
    num_col_patches = (model.config.image_width - po) // (pw - po)
    freq_pos = tf.tile(model.freq_position_embedding, [1, 1, num_col_patches, 1])
    freq_pos = tf.reshape(freq_pos, [1, num_row_patches * num_col_patches, model.config.hidden_dim])
    x = x + freq_pos

    # Add CLS
    cls_tokens = tf.broadcast_to(model.cls_token, [tf.shape(x)[0], 1, model.config.hidden_dim])
    x = tf.concat([cls_tokens, x], axis=1)

    # Through transformer blocks
    if block_index is None:
        for block in model.transformer_blocks:
            x = block(x, training=False)
        return x
    else:
        block_index = int(block_index)
        assert 0 <= block_index < len(model.transformer_blocks), "Invalid block_index"
        for i, block in enumerate(model.transformer_blocks):
            x = block(x, training=False)
            if i == block_index:
                break
        return x


def _hf_vit_ti_tokens_until_block(model, spec_bvhwc, block_index=None):
    """
    spec_bvhwc: (B, V, H, W, C)
    Returns tokens at block_index (after that layer), shape (B*V, 1+N, D)
    If block_index is None, returns final tokens (post-encoder + layernorm).
    """
    H = model.config.image_height
    W = model.config.image_width
    C = model.config.num_channels

    x = tf.reshape(spec_bvhwc, [-1, H, W, C])  # (B*V, H, W, C)
    # Pad to 256x256 and 3 channels as in model.call
    if W != 256 or H != 256:
        paddings = [[0, 0], [0, 256 - H], [0, 256 - W], [0, 0]]
        x = tf.pad(x, paddings)
    # channel pad
    if x.shape[-1] == 2:
        x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [0, 1]])

    # NCHW for HF embeddings
    x_nchw = tf.transpose(x, [0, 3, 1, 2])
    embedding_output = model.vit.embeddings(pixel_values=x_nchw, training=False)

    if block_index is None:
        encoder_outputs = model.vit.encoder(
            hidden_states=embedding_output,
            head_mask=[None] * model.hf_config.num_hidden_layers,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            training=False,
        )
        seq = encoder_outputs.last_hidden_state
        seq = model.vit.layernorm(seq, training=False)
        return seq
    else:
        enc_out = model.vit.encoder(
            hidden_states=embedding_output,
            head_mask=[None] * model.hf_config.num_hidden_layers,
            output_attentions=False,
            output_hidden_states=True,
            return_dict=True,
            training=False,
        )
        # hidden_states: tuple(len=num_layers+1?) depending on HF; we take layer index
        # Be robust: if first is embedding, then layers follow. We'll try index+1.
        hs = enc_out.hidden_states
        # Fallback if structure differs
        idx = int(block_index)
        if len(hs) == model.hf_config.num_hidden_layers:
            tokens = hs[idx]
        else:
            # Expect embeddings at hs[0], layers at 1..L
            tokens = hs[idx + 1]
        return tokens


def deep_layer_pca(spec, model, block_index=None, smooth_sigma=0.0):
    """
    Deep layer PCA over tokens (excluding CLS), reshaped to grid.
    spec: (H, W, C) in [0,1]
    block_index: int or None (None = last)
    Returns:
      - rgb_big: (H, W, 3) in [0,1]
      - rgb_small: (Hr, Wr, 3)
      - pca: fitted PCA object
    """
    kind = _detect_model_kind(model)
    x = tf.convert_to_tensor(spec[None, None, ...], dtype=tf.float32)  # (1,1,H,W,C)
    if kind == "custom_vit":
        tokens = _custom_vit_tokens_until_block(model, x, block_index=block_index)  # (1, 1+N, D) with B*V=1
        D = int(tokens.shape[-1])
        # drop CLS
        tok = tokens[:, 1:, :]
        # reshape to grid
        Hr = model.num_row_patch
        Wr = model.num_col_patch
        grid = tf.reshape(tok, [Hr, Wr, D]).numpy()
    else:
        tokens = _hf_vit_ti_tokens_until_block(model, x, block_index=block_index)  # (1, 1+N, D)
        D = int(tokens.shape[-1])
        tok = tokens[:, 1:, :]
        # For ViT-Ti 256/16=16
        Hr, Wr = 16, 16
        grid = tf.reshape(tok, [Hr, Wr, D]).numpy()

    rgb_small, pca = _fit_pca_to_grid(grid, n_components=3, smooth_sigma=smooth_sigma)
    H = int(spec.shape[0])
    W = int(spec.shape[1])
    rgb_big = _upsample_to(rgb_small, (H, W))
    return {"rgb_big": rgb_big, "rgb_small": rgb_small, "pca": pca}


# ================================
# 3) Saliency PCA (gradient-based)
# ================================

def saliency_pca(spec, model, block_index=None, pca_obj=None, smooth_sigma=2.0):
    """
    Gradient saliency guided by PCA directions of deep tokens.
    spec: (H, W, C)
    block_index: which deep layer tokens to use for PCA (None = last)
    pca_obj: optional pre-fitted PCA (with 3 components). If None, will compute from current spec.
    Returns:
      - rgb: (H, W, 3) in [0,1], blurred
      - grads_raw: list of three raw gradient maps before norm
    """
    kind = _detect_model_kind(model)
    # First pass to get PCA components if needed
    if pca_obj is None:
        dl = deep_layer_pca(spec, model, block_index=block_index, smooth_sigma=0.0)
        pca_obj = dl["pca"]

    # Prepare constants for directions (3 principal components)
    P = np.asarray(pca_obj.components_[:3], dtype=np.float32)  # (3, D)
    P_tf = tf.convert_to_tensor(P, dtype=tf.float32)  # (3, D)

    # Compute three separate gradients (per component)
    grads_rgb = []
    for k in range(3):
        with tf.GradientTape() as tape_k:
            xk = tf.convert_to_tensor(spec[None, None, ...], dtype=tf.float32)
            tape_k.watch(xk)
            if kind == "custom_vit":
                tokens_k = _custom_vit_tokens_until_block(model, xk, block_index=block_index)
            else:
                tokens_k = _hf_vit_ti_tokens_until_block(model, xk, block_index=block_index)
            tok_k = tokens_k[:, 1:, :]
            proj_k = tf.linalg.matvec(tok_k, P_tf[k])  # (1, N)
            yk = tf.reduce_sum(tf.square(proj_k))
        gk = tape_k.gradient(yk, xk)[0, 0].numpy()  # (H, W, C)
        # Reduce channel magnitude (stereo) to a single 2D map
        gk = np.abs(gk).mean(axis=-1)  # (H, W)
        grads_rgb.append(gk)

    # Stack as RGB and normalize
    rgb = np.stack(grads_rgb, axis=-1)  # (H, W, 3)
    rgb = _percentile_normalize(rgb, lo=1.0, hi=99.0)
    if smooth_sigma and smooth_sigma > 0:
        rgb = _gaussian_blur_rgb(rgb, sigma=float(smooth_sigma))
    return {"rgb": rgb, "grads_raw": grads_rgb}


# ================================
# Rendering helpers
# ================================

def overlay_on_spectrogram(spec, rgb, alpha=0.45):
    """
    Returns a matplotlib figure with spectrogram background and RGB overlay.
    spec: (H, W, C)
    rgb: (H, W, 3) in [0,1]
    """
    spec_gray = _spectrogram_to_gray(spec)
    spec_gray = _flip_freq_axis(spec_gray)
    rgb_vis = _flip_freq_axis(rgb)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8 / CONFIG.image_width * CONFIG.image_height))
    ax.imshow(spec_gray, cmap="gray", aspect="auto")
    ax.imshow(rgb_vis, alpha=alpha, aspect="auto")
    ax.axis("off")
    return fig


def render_full_panel(spec, model, save_path=None, dense_stride=4, blocks=(5, 11)):
    """
    Renders a 5-panel figure:
      [Input]
      [Dense-grid PCA]
      [Deep-layer PCA @ block blocks[0]]
      [Deep-layer PCA @ block blocks[1]]
      [Saliency PCA (using deeper block)]
    """
    spec_gray = _flip_freq_axis(_spectrogram_to_gray(spec))
    dense = dense_grid_pca(spec, model, stride=dense_stride, smooth_sigma=1.0)
    deep_mid = deep_layer_pca(spec, model, block_index=blocks[0], smooth_sigma=0.5)
    deep_high = deep_layer_pca(spec, model, block_index=blocks[1], smooth_sigma=0.0)
    sal = saliency_pca(spec, model, block_index=blocks[1], pca_obj=deep_high["pca"], smooth_sigma=2.0)

    fig, axes = plt.subplots(1, 5, figsize=(20, 4), constrained_layout=True)
    # Input
    axes[0].imshow(spec_gray, cmap="gray", aspect="auto")
    axes[0].set_title("Input Spectrogram")
    axes[0].axis("off")
    # Dense-grid
    axes[1].imshow(_flip_freq_axis(dense["rgb_big"]), aspect="auto")
    axes[1].set_title("Dense-grid PCA")
    axes[1].axis("off")
    # Deep mid
    axes[2].imshow(_flip_freq_axis(deep_mid["rgb_big"]), aspect="auto")
    axes[2].set_title(f"Deep-layer PCA (block {blocks[0]})")
    axes[2].axis("off")
    # Deep high
    axes[3].imshow(_flip_freq_axis(deep_high["rgb_big"]), aspect="auto")
    axes[3].set_title(f"Deep-layer PCA (block {blocks[1]})")
    axes[3].axis("off")
    # Saliency
    axes[4].imshow(_flip_freq_axis(sal["rgb"]), aspect="auto")
    axes[4].set_title("Saliency PCA")
    axes[4].axis("off")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


# ================================
# Convenience entry-point
# ================================

def visualize_three(spec, model, out_dir="assets", basename="panel", dense_stride=4, blocks=(5, 11)):
    """
    Generate and save:
      - dense_grid.png
      - deep_block_X.png
      - deep_block_Y.png
      - saliency_pca.png
      - full_panel.png
    """
    os.makedirs(out_dir, exist_ok=True)
    dense = dense_grid_pca(spec, model, stride=dense_stride, smooth_sigma=1.0)
    deep_mid = deep_layer_pca(spec, model, block_index=blocks[0], smooth_sigma=0.5)
    deep_high = deep_layer_pca(spec, model, block_index=blocks[1], smooth_sigma=0.0)
    sal = saliency_pca(spec, model, block_index=blocks[1], pca_obj=deep_high["pca"], smooth_sigma=2.0)

    # Save individual overlays
    overlay_on_spectrogram(spec, dense["rgb_big"]).savefig(os.path.join(out_dir, f"{basename}_dense.png"), dpi=200, bbox_inches="tight")
    plt.close()
    overlay_on_spectrogram(spec, deep_mid["rgb_big"]).savefig(os.path.join(out_dir, f"{basename}_deep_{blocks[0]}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    overlay_on_spectrogram(spec, deep_high["rgb_big"]).savefig(os.path.join(out_dir, f"{basename}_deep_{blocks[1]}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    overlay_on_spectrogram(spec, sal["rgb"]).savefig(os.path.join(out_dir, f"{basename}_saliency.png"), dpi=200, bbox_inches="tight")
    plt.close()

    # Save full panel
    render_full_panel(spec, model, save_path=os.path.join(out_dir, f"{basename}_panel.png"), dense_stride=dense_stride, blocks=blocks)
    plt.close("all")


