import streamlit as st
import numpy as np
from PIL import Image, ImageOps, ImageResampling
from streamlit_cropper import st_cropper
import cv2

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='アップロードされた画像', use_column_width=True)

    cropped_image = st_cropper(image, aspect_ratio=(1, 1))
    st.image(cropped_image, caption='トリミングされた画像', use_column_width=True)
    if st.button("レゴブロックの設計図を生成"):
      cropped_image = cropped_image.resize((48, 48), ImageResampling.LANCZOS)
      
      img_array = np.array(cropped_image.convert("RGB"))

      def convert_to_lego_colors(img_array):
          lego_colors = {
              'red': [255, 0, 0],
              'blue': [0, 0, 255],
              'yellow': [255, 255, 0]
          }
          for i in range(img_array.shape[0]):
              for j in range(img_array.shape[1]):
                  pixel = img_array[i, j]
                  distances = {color: np.linalg.norm(pixel - np.array(rgb)) for color, rgb in lego_colors.items()}
                  closest_color = min(distances, key=distances.get)
                  img_array[i, j] = lego_colors[closest_color]
          return img_array
      
      lego_image_array = convert_to_lego_colors(img_array)
      lego_image = Image.fromarray(lego_image_array.astype('uint8'), 'RGB')
      st.image(lego_image, caption='レゴブロックの設計図', use_column_width=True)

      st.download_button(label='設計図をダウンロード', data=lego_image.tobytes(), file_name='lego_design.png', mime='image/png')
