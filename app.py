import io
import cv2
import numpy as np
import streamlit as st
from PIL import Image


st.set_page_config(page_title="Photo Editor", layout="wide")


# ---------------------------
# Helper functions
# ---------------------------
def load_image(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return np.array(image, dtype=np.uint8)


def resize_image(img, width, height):
    return cv2.resize(
        img,
        (int(width), int(height)),
        interpolation=cv2.INTER_AREA
    )


def adjust_brightness_contrast(img, brightness, contrast):
    return cv2.convertScaleAbs(img, alpha=contrast, beta=brightness)


def apply_grayscale(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_blur(img, blur_strength):
    if blur_strength % 2 == 0:
        blur_strength += 1
    return cv2.GaussianBlur(img, (blur_strength, blur_strength), 0)


def apply_sharpen(img):
    kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    return cv2.filter2D(img, -1, kernel)


def apply_warm_filter(img):
    warm = img.copy().astype(np.int16)
    warm[:, :, 0] = np.clip(warm[:, :, 0] + 20, 0, 255)   # Red
    warm[:, :, 1] = np.clip(warm[:, :, 1] + 10, 0, 255)   # Green
    warm[:, :, 2] = np.clip(warm[:, :, 2] - 10, 0, 255)   # Blue
    return warm.astype(np.uint8)


def apply_portrait_blur(img):
    h, w, _ = img.shape

    blurred = cv2.GaussianBlur(img, (41, 41), 0)

    mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (w // 4, h // 3)

    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    mask = cv2.GaussianBlur(mask, (41, 41), 0)

    mask_3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB) / 255.0
    output = (img * mask_3 + blurred * (1 - mask_3)).astype(np.uint8)

    return output


def apply_edge_detection(img, threshold1, threshold2):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, threshold1, threshold2)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)


def apply_sketch_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2RGB)


def apply_cartoon_effect(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray_blur,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        9
    )

    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon


def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)

    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]

    rotated = cv2.warpAffine(img, matrix, (new_w, new_h))
    return rotated


def get_image_download_bytes(img):
    buffer = io.BytesIO()
    Image.fromarray(img).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------
# Main app
# ---------------------------
st.title("Photo Editor using OpenCV & Streamlit")
st.write("Upload → Adjust → Apply Filters/Effects → View → Download")

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    original_image = load_image(uploaded_file)
    h, w, _ = original_image.shape

    st.sidebar.header("Basic Adjustments")

    min_width = min(150, int(w))
    min_height = min(200, int(h))

    width = st.sidebar.slider("Width", min_width, int(w), int(w))
    height = st.sidebar.slider("Height", min_height, int(h), int(h))

    brightness = st.sidebar.slider("Brightness", -100, 100, 0)
    contrast = st.sidebar.slider("Contrast", 0.5, 3.0, 1.0)

    processed_image = resize_image(original_image, width, height)
    processed_image = adjust_brightness_contrast(
        processed_image,
        brightness,
        contrast
    )

    st.sidebar.header("Filters")
    filter_option = st.sidebar.selectbox(
        "Choose a filter",
        [
            "None",
            "Grayscale",
            "Blur",
            "Sharpen",
            "Warm",
            "Portrait Blur"
        ]
    )

    if filter_option == "Grayscale":
        processed_image = apply_grayscale(processed_image)

    elif filter_option == "Blur":
        blur_strength = st.sidebar.slider("Blur Strength", 1, 25, 7, step=2)
        processed_image = apply_blur(processed_image, blur_strength)

    elif filter_option == "Sharpen":
        processed_image = apply_sharpen(processed_image)

    elif filter_option == "Warm":
        processed_image = apply_warm_filter(processed_image)

    elif filter_option == "Portrait Blur":
        processed_image = apply_portrait_blur(processed_image)

    st.sidebar.header("Extra Features")
    extra_option = st.sidebar.selectbox(
        "Choose an extra feature",
        [
            "None",
            "Edge Detection",
            "Sketch Effect",
            "Cartoon Effect",
            "Rotate"
        ]
    )

    if extra_option == "Edge Detection":
        threshold1 = st.sidebar.slider("Threshold 1", 0, 255, 100)
        threshold2 = st.sidebar.slider("Threshold 2", 0, 255, 200)
        processed_image = apply_edge_detection(processed_image, threshold1, threshold2)

    elif extra_option == "Sketch Effect":
        processed_image = apply_sketch_effect(processed_image)

    elif extra_option == "Cartoon Effect":
        processed_image = apply_cartoon_effect(processed_image)

    elif extra_option == "Rotate":
        angle = st.sidebar.slider("Rotation Angle", -180, 180, 0)
        processed_image = rotate_image(processed_image, angle)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")
        st.image(original_image, use_container_width=True)

    with col2:
        st.subheader("Edited Image")
        st.image(processed_image, width=width)

    download_data = get_image_download_bytes(processed_image)

    st.download_button(
        label="Download Edited Image",
        data=download_data,
        file_name="edited_image.png",
        mime="image/png"
    )

else:
    st.info("Upload an image to start editing.")