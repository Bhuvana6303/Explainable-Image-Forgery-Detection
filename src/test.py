from inference import predict_image

img_path = "/Users/venkatsomanadh/Downloads/Forgery Detection/dataset/original_images/CRW_4809_scale.jpg"

result = predict_image(img_path)
print("Prediction:", result)
