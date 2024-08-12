import streamlit as st

st.title("🎈 My new app")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)
import streamlit as st
import cv2
import numpy as np

st.title("License Plate Recognition")
st.write(
    "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
)

def extract_text_from_image(image):
  """Extracts text from an image using OpenCV and PyTesseract.

  Args:
    image: A numpy array representing the image.

  Returns:
    Extracted text from the image.
  """

  # Preprocess the image (optional)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # Extract text using PyTesseract
  text = pytesseract.image_to_string(thresh)

  return text

def main():
  st.title("Image to Text")

  uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
  if uploaded_file is not None:
      image_data = uploaded_file.read()
      image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
      st.image(image, caption='Uploaded Image', use_column_width=True)

      if st.button("Extract Text"):
        text = extract_text_from_image(image)
        st.text_area("Extracted Text:", value=text)

if __name__ == "__main__":
  main()


