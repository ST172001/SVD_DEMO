from io import BytesIO
import streamlit as st
from PIL import Image
import numpy as np
from svd import compute_svd, get_approximate_image

st.title("Welcome to the SVD Demo!!")
file = st.file_uploader("Upload file", type=["png","jpeg"], key="user_image")
# st.write("Or select a default image")
# Example images - replace these with your actual images or file paths
image_paths = ['doraemon.png', 'lena.png', 'pizza.png', 'Flag_of_India.png']
col1, col2 = st.columns(2)
with col1:
    if file is not None:
        image = Image.open(file)
        max_size = 300
        if image.size[0] > max_size or image.size[1] > max_size:
            ratio = min(max_size / image.size[0], max_size / image.size[1])
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            st.write(
                f"Your uploaded image has been resized to {new_size[0]}*{new_size[1]} px"
            )
            image = image.resize(new_size).convert("L")
        else:
            st.write(
                f"Your uploaded image is of the size {image.size[0]}*{image.size[1]} px"
            )
            image = image.convert("L")
        for i in range(7):
            st.write("")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        # st.write(
        #     f"The number of bytes required to store the black and white original image is {image.size[0]*image.size[1]} bytes."
        # )
        image_np = np.array(image)
        eigen_values, right_vs = compute_svd(image_np)

with col2:
    if file is not None:

        st.write(f"The number of singular values is {len(eigen_values)}")
        number_of_top_singular_values = st.slider(
            "Select the number of singular values (Move the seek until the right image becomes similar to the left image)",
            min_value=0,
            max_value=len(eigen_values),
            step=1,
        )
        Y = np.abs(
            get_approximate_image(
                image_np, number_of_top_singular_values, np.abs(eigen_values), right_vs
            )
        )
        if (Y.max() - Y.min()) != 0:
            Y_normalized = (Y - Y.min()) / (Y.max() - Y.min()) * 255
            Y_normalized = Y_normalized.astype(np.uint8)
        else:
            Y_normalized = Y.astype(np.uint8)

        # Create the PIL image
        image_pil = Image.fromarray(Y_normalized)

        # Display the reconstructed image
        st.image(
            image_pil,
            caption=f"Reconstructed Image with {number_of_top_singular_values} singular values",
            use_column_width=True,
        )
        # st.write(
        #     f"The number of bytes required to store the approximate SVD image with {number_of_top_singular_values} singular values is approximately {(Y.shape[0]+Y.shape[1])*number_of_top_singular_values} bytes."
        # )
        original_size = image.size[0] * image.size[1]
        new_image_size = (Y.shape[0] + Y.shape[1]) * number_of_top_singular_values
        percentage_reduction = ((original_size - new_image_size) * 100) / original_size
        # st.markdown(
        #     f"<p style='color:green;'>Compression: {percentage_reduction:.2f}% &#x2193;</p>",
        #     unsafe_allow_html=True,
        # )
