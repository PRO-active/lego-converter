import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from streamlit_cropper import st_cropper

COLOR_SETS = {
  "赤, 青, 黄": {
    'red': [255, 0, 0],
    'blue': [0, 0, 255],
    'yellow': [255, 255, 0]
  },
  "5色": {
    'yellow': [255, 255, 0],
    'white': [255, 255, 255],
    'light_gray': [192, 192, 192],
    'dark_gray': [96, 96, 96],
    'black': [0, 0, 0]
  },
  "色の制限なし": None
}

COLOR_WEIGHTS = {
  'red': 0.2,
  'blue': 0.2,
  'yellow': 0.2,
  'white': 0.2,
  'light_gray': 0.2,
  'dark_gray': 0.2,
  'black': 0.2
}

def select_color(pixel_value, color_set):
  distances = {color: np.abs(pixel_value - np.mean(rgb)) * (1 + COLOR_WEIGHTS[color]) for color, rgb in color_set.items()}
  closest_color = min(distances, key=distances.get)
  return closest_color

def convert_to_lego_colors(img_array, color_set):
  gray_image = ImageOps.grayscale(Image.fromarray(img_array))
  gray_array = np.array(gray_image)
  lego_image_array = np.zeros((48, 48, 3), dtype=np.uint8)

  for i in range(img_array.shape[0]):
    for j in range(img_array.shape[1]):
      pixel_value = gray_array[i, j]
      closest_color = select_color(pixel_value, color_set)
      lego_image_array[i, j] = color_set[closest_color]

  return lego_image_array

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
  image = Image.open(uploaded_file)
  st.image(image, caption='アップロードされた画像', use_column_width=True)

  cropped_image = st_cropper(image, aspect_ratio=(1, 1))
  st.image(cropped_image, caption='トリミングされた画像', use_column_width=True)

  color_set_name = st.selectbox("変換後の色を選択してください", options=list(COLOR_SETS.keys()))
  selected_color_set = COLOR_SETS[color_set_name]

  if st.button("レゴブロックの設計図を生成"):
    cropped_image = cropped_image.resize((48, 48), Image.LANCZOS)
    
    img_array = np.array(cropped_image.convert("RGB"))

    if selected_color_set is None:
      lego_image_array = img_array
    else:
      lego_image_array = convert_to_lego_colors(img_array, selected_color_set)

    lego_image = Image.fromarray(lego_image_array.astype('uint8'), 'RGB')
    st.image(lego_image, caption='レゴブロックの設計図', use_column_width=True)

    st.download_button(label='設計図をダウンロード', data=lego_image.tobytes(), file_name='lego_design.png', mime='image/png')


