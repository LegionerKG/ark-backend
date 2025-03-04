from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI(title="SmartBiz 2.0")

# Хранилище данных (временное, для теста)
uploaded_data = None

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Загрузка Excel/CSV и возврат первых строк для проверки"""
    global uploaded_data
    try:
        # Чтение файла (CSV или Excel)
        if file.filename.endswith(".csv"):
            uploaded_data = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            uploaded_data = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        # Возвращаем первые 5 строк для подтверждения
        return JSONResponse(content={"data": uploaded_data.head().to_dict(orient="records")})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/chart")
async def get_chart(column: str = "sales"):
    """Построение графика по указанному столбцу (по умолчанию 'sales')"""
    global uploaded_data
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet. Please upload a file first.")
    
    # Проверка наличия столбца
    if column not in uploaded_data.columns:
        raise HTTPException(status_code=400, detail=f"Column '{column}' not found in data.")
    
    # Построение графика
    plt.figure(figsize=(10, 6))
    plt.plot(uploaded_data.index, uploaded_data[column], marker="o")
    plt.title(f"{column.capitalize()} Trend")
    plt.xlabel("Index")
    plt.ylabel(column.capitalize())
    plt.grid(True)
    
    # Сохранение графика в base64
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    return JSONResponse(content={"image": f"data:image/png;base64,{img_base64}"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
