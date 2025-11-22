import pandas as pd
import requests

BASE_URL  = "http://34.122.51.94:9000"
TOKEN   = "Your-API-TOKEN-Here"  # replace with your actual API key

TASK_ID   = "08-watermark-detection"


df = pd.DataFrame({
    "image_name": image_names,
    "label": predictions,
})
df.to_csv("submission.csv", index=None)
response = requests.post(
    "http://34.122.51.94:9000",
    files={"file": open("submission.csv", "rb")},
    headers={"token": "TOKEN"}
)
print(response.json())
