from inference_sdk import InferenceHTTPClient

CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="UJpHlPdOWnAAVJ46BfwN"
)

result = CLIENT.infer("D:\plastiphalt_ai_project\plastiphalt_ai\pothole_detection\sample_image_1.jpeg", model_id="pothole-vhmow/2")
print(result)
