from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pyngrok import ngrok
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import nest_asyncio
import uvicorn

# Cài đặt nest_asyncio để hỗ trợ chạy trên Google Colab
nest_asyncio.apply()

# Tải các mô hình đã huấn luyện
model_lab = load_model('/content/drive/MyDrive/TunaAnn_Data/model_ann_lab.h5', custom_objects={'mse': MeanSquaredError()})
model_rgb = load_model('/content/drive/MyDrive/TunaAnn_Data/model_ann_rgb.h5', custom_objects={'mse': MeanSquaredError()})

# Tạo FastAPI app
app = FastAPI()

# Hàm dự đoán
def du_doan_chi_so(dau_vao, mode='lab'):
    dau_vao = np.array([dau_vao])  # Chuyển thành array để đưa vào mô hình
    if mode == 'lab':
        pred = model_lab.predict(dau_vao)[0]
    elif mode == 'rgb':
        pred = model_rgb.predict(dau_vao)[0]
    else:
        raise ValueError("Mode phải là 'lab' hoặc 'rgb'")
    return {
        'MetMb': round(float(pred[0]), 2),
        'TBARS': round(float(pred[1]), 2),
        'Peroxide': round(float(pred[2]), 2)
    }

# Tạo model yêu cầu dữ liệu đầu vào
class InputData(BaseModel):
    lab_input: list
    mode: str

# Route cho trang chủ
@app.get("/")
def read_root():
    return {"message": "API dự đoán chất lượng cá ngừ - Sử dụng POST /predict để nhận dự đoán"}

# API route dự đoán
@app.post("/predict")
async def predict(data: InputData):
    # Kiểm tra xem đầu vào có hợp lệ không
    if not data.lab_input or len(data.lab_input) != 3:
        raise HTTPException(status_code=400, detail="lab_input phải có 3 giá trị")
    
    lab_input = data.lab_input
    mode = data.mode

    # Kiểm tra mode hợp lệ
    if mode not in ['lab', 'rgb']:
        raise HTTPException(status_code=400, detail="Mode phải là 'lab' hoặc 'rgb'")

    # Dự đoán
    result = du_doan_chi_so(lab_input, mode)
    return result


