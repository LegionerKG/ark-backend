from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import json  # Добавлен импорт json
from typing import Optional
import os

app = FastAPI(title="SmartBiz 2.0")

# Хранилище данных и маппинга столбцов
uploaded_data = None
column_mapping = {
    "revenue": None,
    "profit": None,
    "date": None,
    "rating": None,
    "discount": None
}

# Настройка OpenAI API
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    revenue: Optional[str] = None,
    profit: Optional[str] = None,
    date: Optional[str] = None,
    rating: Optional[str] = None,
    discount: Optional[str] = None
):
    global uploaded_data, column_mapping
    try:
        if file.filename.endswith(".csv"):
            uploaded_data = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            uploaded_data = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        user_mapping = {"revenue": revenue, "profit": profit, "date": date, "rating": rating, "discount": discount}
        if any(user_mapping.values()):
            for key, value in user_mapping.items():
                if value and value in uploaded_data.columns:
                    column_mapping[key] = value
                elif value:
                    raise HTTPException(status_code=400, detail=f"Column '{value}' not found in data.")
            message = "Columns updated with user input."
        else:
            column_mapping = {key: None for key in column_mapping}
            for col in uploaded_data.columns:
                if pd.api.types.is_datetime64_any_dtype(uploaded_data[col]) or "date" in col.lower():
                    column_mapping["date"] = col
                elif pd.api.types.is_numeric_dtype(uploaded_data[col]):
                    if column_mapping["revenue"] is None and uploaded_data[col].mean() > 1000:
                        column_mapping["revenue"] = col
                    elif column_mapping["profit"] is None and (column_mapping["revenue"] is None or uploaded_data[col].mean() < uploaded_data[column_mapping["revenue"]].mean()):
                        column_mapping["profit"] = col
                    elif "rating" in col.lower() or uploaded_data[col].max() <= 5:
                        column_mapping["rating"] = col
                    elif "discount" in col.lower() or "%" in col:
                        column_mapping["discount"] = col
            
            unmapped = [key for key, value in column_mapping.items() if value is None]
            if unmapped:
                message = (
                    f"Could not auto-detect: {unmapped}. Please re-upload with parameters like "
                    f"'revenue=Sales', 'profit=Profit', etc., to specify columns."
                )
            else:
                message = "Columns auto-detected successfully."
        
        if column_mapping["date"]:
            uploaded_data[column_mapping["date"]] = pd.to_datetime(uploaded_data[column_mapping["date"]])
        
        data_preview = uploaded_data.head().to_dict(orient="records")
        for row in data_preview:
            if column_mapping["date"] and isinstance(row[column_mapping["date"]], pd.Timestamp):
                row[column_mapping["date"]] = row[column_mapping["date"]].isoformat()
        
        return JSONResponse(content={
            "data": data_preview,
            "mapping": column_mapping,
            "message": message
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/chart")
async def get_chart(column: Optional[str] = None):
    global uploaded_data, column_mapping
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")
    
    col = column if column else column_mapping["revenue"]
    if not col or col not in uploaded_data.columns:
        raise HTTPException(status_code=400, detail=f"Column '{col}' not found or not defined.")
    
    plt.figure(figsize=(10, 6))
    if column_mapping["date"]:
        plt.plot(uploaded_data[column_mapping["date"]], uploaded_data[col], marker="o")
        plt.xlabel("Date")
    else:
        plt.plot(uploaded_data.index, uploaded_data[col], marker="o")
        plt.xlabel("Index")
    plt.title(f"{col.capitalize()} Trend")
    plt.ylabel(col.capitalize())
    plt.grid(True)
    plt.xticks(rotation=45)
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close()
    
    return JSONResponse(content={"image": f"data:image/png;base64,{img_base64}"})

@app.get("/metrics")
async def get_metrics():
    global uploaded_data, column_mapping
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")
    
    data = uploaded_data
    metrics = {}

    if column_mapping["revenue"]:
        revenue_col = column_mapping["revenue"]
        metrics["total_revenue"] = float(data[revenue_col].sum())
        metrics["average_revenue"] = float(data[revenue_col].mean())
    
    if column_mapping["profit"]:
        profit_col = column_mapping["profit"]
        metrics["total_profit"] = float(data[profit_col].sum())
        metrics["average_profit"] = float(data[profit_col].mean())
        if column_mapping["revenue"]:
            metrics["profit_margin_percent"] = round((metrics["total_profit"] / metrics["total_revenue"]) * 100, 2) if metrics["total_revenue"] > 0 else 0

    if column_mapping["date"] and column_mapping["revenue"]:
        data["Month"] = data[column_mapping["date"]].dt.to_period("M")
        monthly_revenue = data.groupby("Month")[column_mapping["revenue"]].sum()
        revenue_trend = monthly_revenue.pct_change().mean() * 100
        metrics["revenue_trend_percent"] = round(revenue_trend, 2) if not pd.isna(revenue_trend) else 0
        metrics["monthly_revenue"] = {str(k): float(v) for k, v in monthly_revenue.to_dict().items()}

    if column_mapping["rating"]:
        metrics["average_rating"] = round(float(data[column_mapping["rating"]].mean()), 2)

    if column_mapping["discount"]:
        metrics["average_discount_percent"] = round(float(data[column_mapping["discount"]].mean()), 2)

    if not metrics:
        raise HTTPException(status_code=400, detail="No relevant columns defined.")
    
    return JSONResponse(content=metrics)

@app.get("/insights")
async def get_insights():
    global uploaded_data, column_mapping
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")
    
    # Подготовка данных для OpenAI
    try:
        data_summary = uploaded_data.describe().to_string()
        metrics_response = await get_metrics()  # Получаем JSONResponse
        metrics = json.loads(metrics_response.body.decode('utf-8'))  # Декодируем в словарь
        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        prompt = (
            f"Analyze the following business data summary and metrics:\n\n"
            f"Data Summary:\n{data_summary}\n\n"
            f"Metrics:\n{metrics_str}\n\n"
            f"Provide detailed insights, identify anomalies, key trends, and actionable recommendations."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data for AI: {str(e)}")
    
    # Запрос к OpenAI API
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.RequestException as e:
        error_detail = f"Error querying OpenAI: {str(e)}"
        if hasattr(e.response, "text"):
            error_detail += f" - {e.response.text}"
        raise HTTPException(status_code=500, detail=error_detail)
    
    return JSONResponse(content={"insights": ai_response})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
