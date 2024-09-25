import numpy as np
import streamlit as st
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image, ImageDraw
import cv2


def get_ellipse_coords(point: tuple[int, int]) -> tuple[int, int, int, int]:
    center = point
    radius = 2
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


@st.cache_data
def get_image(file):
    img = Image.open(file)
    width = 512
    height = img.height * width // img.width

    img = img.resize((width, height))

    return img


st.set_page_config(layout="wide")

img_columns = st.columns(2)

with img_columns[0]:
    st.write("### Image 1")
    img_file_1 = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="img_1")

with img_columns[1]:
    st.write("### Image 2")
    img_file_2 = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"], key="img_2")

if img_file_1 is None or img_file_2 is None:
    st.stop()

with img_columns[0]:
    img_1 = get_image(img_file_1)
    draw_1 = ImageDraw.Draw(img_1)

    if "points_1" not in st.session_state:
        st.session_state["points_1"] = []

    for point in st.session_state["points_1"]:
        coords = get_ellipse_coords(point)
        draw_1.ellipse(coords, fill="red")

    value_1 = streamlit_image_coordinates(img_1, key="pil_1")

    if value_1 is not None:
        point = value_1["x"], value_1["y"]

        if point not in st.session_state["points_1"]:
            st.session_state["points_1"].append(point)
            st.rerun()

with img_columns[1]:
    img_2 = get_image(img_file_2)
    draw_2 = ImageDraw.Draw(img_2)

    if "points_2" not in st.session_state:
        st.session_state["points_2"] = []

    for point in st.session_state["points_2"]:
        coords = get_ellipse_coords(point)
        draw_2.ellipse(coords, fill="red")

    value_2 = streamlit_image_coordinates(img_2, key="pil_2")

    if value_2 is not None:
        point = value_2["x"], value_2["y"]

        if point not in st.session_state["points_2"]:
            st.session_state["points_2"].append(point)
            st.rerun()

align_imgs = st.button("Align images")

# based on selected points on both images, calculate the transformation matrix
if align_imgs:
    with st.spinner("Aligning images..."):
        # get points from both images
        points_1 = st.session_state["points_1"]
        points_2 = st.session_state["points_2"]

        # calculate transformation matrix
        matrix, _ = cv2.estimateAffinePartial2D(
            from_=np.array([points_1]),
            to=np.array([points_2]),
            method=cv2.RANSAC,
            ransacReprojThreshold=5,
        )

        # apply transformation to the first image
        img_1_aligned = cv2.warpAffine(
            cv2.cvtColor(np.array(img_1), cv2.COLOR_RGB2BGR),
            matrix,
            (img_2.width, img_2.height),
        )

        img_1_aligned = Image.fromarray(cv2.cvtColor(img_1_aligned, cv2.COLOR_BGR2RGB))

        # display the aligned images
        st.image([img_1_aligned, img_2], width=512)

        # display img_1_aligned over b&w img_2
        img_2_gray = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2GRAY)
        img_2_gray = cv2.cvtColor(img_2_gray, cv2.COLOR_GRAY2RGB)

        img_1_aligned = cv2.cvtColor(np.array(img_1_aligned), cv2.COLOR_RGB2BGR)
        img_1_aligned_gray = cv2.cvtColor(img_1_aligned, cv2.COLOR_BGR2GRAY)

        _, mask = cv2.threshold(img_1_aligned_gray, 1, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        img_2_bg = cv2.bitwise_and(img_2_gray, img_2_gray, mask=mask_inv)
        img_1_fg = cv2.bitwise_and(img_1_aligned, img_1_aligned, mask=mask)

        result = cv2.add(img_2_bg, img_1_fg)
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)

        st.image(result, width=512)
