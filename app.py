import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# ---------------- Helper Functions ----------------
def entropic_threshold(hist):
    hist = hist.astype(np.float32)
    hist /= hist.sum()
    cumsum = np.cumsum(hist)
    entropy_total = np.zeros_like(hist)

    for T in range(1, len(hist)):
        p0 = hist[:T]
        c0 = cumsum[T - 1]
        entropy_bkg = -np.sum((p0 / c0) * np.log(p0 / c0 + 1e-12)) if c0 > 0 else 0
        p1 = hist[T:]
        c1 = 1.0 - c0
        entropy_fgd = -np.sum((p1 / c1) * np.log(p1 / c1 + 1e-12)) if c1 > 0 else 0
        entropy_total[T] = entropy_bkg + entropy_fgd

    return np.argmax(entropy_total)

def compute_patch_thresholds(image, block_size):
    h, w = image.shape
    h_blocks = h // block_size[0]
    w_blocks = w // block_size[1]

    thresholds = []
    segmented = np.zeros_like(image)

    for i in range(h_blocks):
        row_thresholds = []
        for j in range(w_blocks):
            y0, y1 = i * block_size[0], (i + 1) * block_size[0]
            x0, x1 = j * block_size[1], (j + 1) * block_size[1]
            patch = image[y0:y1, x0:x1]

            hist = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            T = entropic_threshold(hist)
            row_thresholds.append(T)

            binary_patch = (patch > T).astype(np.uint8) * 255
            segmented[y0:y1, x0:x1] = binary_patch
        thresholds.append(row_thresholds)

    return thresholds, segmented

def global_thresholding(image):
    T, binary = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return T, binary

def draw_patch_grid(image, block_size, thresholds=None):
    img_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = image.shape
    h_blocks = h // block_size[0]
    w_blocks = w // block_size[1]

    for i in range(h_blocks):
        for j in range(w_blocks):
            y0, y1 = i * block_size[0], (i + 1) * block_size[0]
            x0, x1 = j * block_size[1], (j + 1) * block_size[1]
            cv2.rectangle(img_color, (x0, y0), (x1-1, y1-1), (0, 255, 0), 1)

            if thresholds:
                T = thresholds[i][j]
                cv2.putText(img_color, f"T={T}", (x0+3, y0+12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)

    return img_color

def show_patches_histograms(image, thresholds, block_size):
    h, w = image.shape
    h_blocks = h // block_size[0]
    w_blocks = w // block_size[1]

    fig, axes = plt.subplots(h_blocks, w_blocks * 2, figsize=(w_blocks * 4, h_blocks * 2.5))
    axes = np.array(axes).reshape((h_blocks, w_blocks * 2))

    for i in range(h_blocks):
        for j in range(w_blocks):
            y0, y1 = i * block_size[0], (i + 1) * block_size[0]
            x0, x1 = j * block_size[1], (j + 1) * block_size[1]
            patch = image[y0:y1, x0:x1]
            hist = cv2.calcHist([patch], [0], None, [256], [0, 256]).flatten()
            T = thresholds[i][j]

            axes[i, j*2].imshow(patch, cmap='gray')
            axes[i, j*2].set_title(f"Patch [{i},{j}]\nT={T}")
            axes[i, j*2].axis('off')

            axes[i, j*2 + 1].plot(hist, color='black')
            axes[i, j*2 + 1].axvline(x=T, color='red', linestyle='--')
            axes[i, j*2 + 1].set_title("Histogram")

    plt.tight_layout()
    st.pyplot(fig)

# ---------------- Streamlit UI ----------------
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Theory", "Implementation"])

if app_mode == "Theory":
    st.title("Theory of Dynamic Entropic Thresholding")
    st.markdown("""
    ### What is Dynamic Thresholding?
    Dynamic (or Adaptive) Thresholding divides an image into patches and computes local thresholds.

    ### What is Entropic Thresholding?
    Entropic Thresholding uses **Shannon Entropy** to find the optimal threshold by maximizing the sum of background and foreground entropy.

    ### Why Convert RGB to Grayscale?
    Thresholding requires a single intensity value per pixel. Therefore, images are converted to grayscale or a brightness channel.

    ### Pipeline Steps:
    1. Upload RGB image.
    2. Convert to Grayscale.
    3. Divide into patches.
    4. Compute entropy-based thresholds for each patch.
    5. Apply thresholds and segment the image.
    6. Visualize patches, histograms, and final output.
    """)
elif app_mode == "Implementation":
    st.title("Dynamic vs Global Thresholding")

    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    patch_size = st.sidebar.slider("Patch Size", min_value=16, max_value=128, step=16, value=64)

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to grayscale
        img_np = np.array(image)
        if img_np.ndim == 3:
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        else:
            img_gray = img_np

        img_gray = cv2.resize(img_gray, (256, 256))

        block_size = (patch_size, patch_size)

        thresholds, dynamic_output = compute_patch_thresholds(img_gray, block_size)
        img_with_grid = draw_patch_grid(img_gray, block_size, thresholds)

        _, global_output = global_thresholding(img_gray)

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Show Patches, Histograms, Thresholds"):
                show_patches_histograms(img_gray, thresholds, block_size)

        with col2:
            if st.button("Show Full Segmentation Pipeline"):
                st.subheader("Grayscale Image")
                st.image(img_gray, caption="Grayscale Image", use_column_width=True)

                st.subheader("Patch Grid")
                st.image(cv2.cvtColor(img_with_grid, cv2.COLOR_BGR2RGB), caption="Patch Grid", use_column_width=True)

                st.subheader("Segmentation: Global vs Dynamic")
                compare = np.concatenate([global_output, dynamic_output], axis=1)
                st.image(compare, caption="Left: Global (Otsu), Right: Dynamic (Entropic)", use_column_width=True)
