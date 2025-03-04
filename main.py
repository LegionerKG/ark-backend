from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests
import json
from typing import Optional
import os

app = FastAPI(title="SmartBiz 2.0")

uploaded_data = None
column_mapping = {
    "revenue": None,
    "profit": None,
    "date": None,
    "rating": None,
    "discount": None,
    "quantity": None,
    "category": None,
    "customer_id": None  # Для service и других типов
}

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key-here")
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

@app.post("/upload")
async def upload_file(
    file: UploadFile = File(...),
    revenue: Optional[str] = None,
    profit: Optional[str] = None,
    date: Optional[str] = None,
    rating: Optional[str] = None,
    discount: Optional[str] = None,
    quantity: Optional[str] = None,
    category: Optional[str] = None,
    customer_id: Optional[str] = None,
    business_type: Optional[str] = "general"
):
    global uploaded_data, column_mapping
    try:
        if file.filename.endswith(".csv"):
            uploaded_data = pd.read_csv(file.file)
        elif file.filename.endswith((".xlsx", ".xls")):
            uploaded_data = pd.read_excel(file.file)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use CSV or Excel.")
        
        uploaded_data.business_type = business_type.lower()
        
        user_mapping = {
            "revenue": revenue, "profit": profit, "date": date, "rating": rating,
            "discount": discount, "quantity": quantity, "category": category, "customer_id": customer_id
        }
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
                    elif "rating" in col.lower() or uploaded_data[col].max() <= 10:
                        column_mapping["rating"] = col
                    elif "discount" in col.lower() or "%" in col:
                        column_mapping["discount"] = col
                    elif "quantity" in col.lower():
                        column_mapping["quantity"] = col
                elif "category" in col.lower() or "product" in col.lower():
                    column_mapping["category"] = col
                elif "customer" in col.lower() or "id" in col.lower():
                    column_mapping["customer_id"] = col
            
            unmapped = [key for key, value in column_mapping.items() if value is None]
            if unmapped:
                message = f"Could not auto-detect: {unmapped}. Please re-upload with parameters."
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
            "business_type": business_type,
            "message": message
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

@app.get("/metrics")
async def get_metrics():
    global uploaded_data, column_mapping
    if uploaded_data is None:
        raise HTTPException(status_code=400, detail="No data uploaded yet.")
    
    data = uploaded_data
    business_type = getattr(data, "business_type", "general")
    metrics = {}
    
    if column_mapping["revenue"]:
        revenue_col = column_mapping["revenue"]
        metrics["total_revenue"] = float(data[revenue_col].sum())
        metrics["average_revenue"] = float(data[revenue_col].mean())
    
    if business_type == "cafe":
        if column_mapping["rating"]:
            metrics["average_rating"] = round(float(data[column_mapping["rating"]].mean()), 2)
            if column_mapping["date"]:
                monthly_rating = data.groupby(data[column_mapping["date"]].dt.to_period("M"))[column_mapping["rating"]].mean()
                metrics["rating_trend_percent"] = round(monthly_rating.pct_change().mean() * 100, 2) if not pd.isna(monthly_rating.pct_change().mean()) else 0
        if column_mapping["date"] and column_mapping["revenue"]:
            data["Day"] = data[column_mapping["date"]].dt.day_name()
            daily_revenue = data.groupby("Day")[column_mapping["revenue"]].mean()
            metrics["daily_revenue"] = {k: float(v) for k, v in daily_revenue.to_dict().items()}
            metrics["peak_day"] = daily_revenue.idxmax()
            data["Hour"] = data[column_mapping["date"]].dt.hour
            hourly_revenue = data.groupby("Hour")[column_mapping["revenue"]].sum()
            metrics["highest_revenue_hour"] = int(hourly_revenue.idxmax()) if not hourly_revenue.empty else None
        if column_mapping["customer_id"]:
            metrics["revenue_per_customer"] = round(metrics["total_revenue"] / data[column_mapping["customer_id"]].nunique(), 2)
    
    elif business_type == "shop":
        if column_mapping["profit"]:
            profit_col = column_mapping["profit"]
            metrics["total_profit"] = float(data[profit_col].sum())
            metrics["average_profit"] = float(data[profit_col].mean())
            if column_mapping["revenue"]:
                metrics["profit_margin_percent"] = round((metrics["total_profit"] / metrics["total_revenue"]) * 100, 2) if metrics["total_revenue"] > 0 else 0
        if "Branch" in data.columns and column_mapping["revenue"]:
            branch_revenue = data.groupby("Branch")[column_mapping["revenue"]].sum()
            metrics["revenue_by_branch"] = {k: float(v) for k, v in branch_revenue.to_dict().items()}
        if column_mapping["quantity"]:
            metrics["total_quantity_sold"] = float(data[column_mapping["quantity"]].sum())
            if column_mapping["date"]:
                days = (data[column_mapping["date"]].max() - data[column_mapping["date"]].min()).days + 1
                metrics["sales_velocity"] = round(metrics["total_quantity_sold"] / days, 2) if days > 0 else 0
        if column_mapping["category"] and column_mapping["quantity"]:
            category_sales = data.groupby(column_mapping["category"])[column_mapping["quantity"]].sum()
            metrics["top_selling_category"] = category_sales.idxmax()
        if "Branch" in data.columns and column_mapping["rating"]:
            branch_rating = data.groupby("Branch")[column_mapping["rating"]].mean()
            metrics["branch_rating_variance"] = round(float(branch_rating.var()), 2) if len(branch_rating) > 1 else 0
    
    elif business_type == "online":
        if column_mapping["date"] and column_mapping["revenue"]:
            data["Month"] = data[column_mapping["date"]].dt.to_period("M")
            monthly_revenue = data.groupby("Month")[column_mapping["revenue"]].sum()
            revenue_trend = monthly_revenue.pct_change().mean() * 100
            metrics["revenue_trend_percent"] = round(revenue_trend, 2) if not pd.isna(revenue_trend) else 0
            metrics["monthly_revenue"] = {str(k): float(v) for k, v in monthly_revenue.to_dict().items()}
            daily_orders = data.groupby(data[column_mapping["date"]].dt.date).size()
            metrics["peak_order_day"] = str(daily_orders.idxmax()) if not daily_orders.empty else None
        if column_mapping["revenue"]:
            metrics["average_order_value"] = metrics["average_revenue"]
        if column_mapping["quantity"]:
            metrics["items_per_order"] = round(float(data[column_mapping["quantity"]].mean()), 2)
        if column_mapping["discount"] and column_mapping["revenue"]:
            metrics["discount_impact"] = round(float(data[column_mapping["discount"]].corr(data[column_mapping["revenue"]])), 2)
        if column_mapping["customer_id"]:
            total_orders = data[column_mapping["customer_id"]].size
            unique_visits = data[column_mapping["customer_id"]].nunique()  # Предполагаем визиты = уникальные клиенты
            metrics["conversion_rate"] = round((total_orders / unique_visits) * 100, 2) if unique_visits > 0 else 0
    
    elif business_type == "restaurant":
        if column_mapping["rating"]:
            metrics["average_rating"] = round(float(data[column_mapping["rating"]].mean()), 2)
            if column_mapping["category"]:
                category_rating = data.groupby(column_mapping["category"])[column_mapping["rating"]].mean()
                metrics["category_rating"] = {k: round(float(v), 2) for k, v in category_rating.to_dict().items()}
        if column_mapping["category"] and column_mapping["revenue"]:
            category_revenue = data.groupby(column_mapping["category"])[column_mapping["revenue"]].sum()
            metrics["revenue_by_category"] = {k: float(v) for k, v in category_revenue.to_dict().items()}
        if column_mapping["revenue"]:
            metrics["average_bill"] = metrics["average_revenue"]
        if column_mapping["date"] and column_mapping["revenue"]:
            data["Day"] = data[column_mapping["date"]].dt.day_name()
            metrics["peak_day"] = data.groupby("Day")[column_mapping["revenue"]].sum().idxmax()
            data["IsWeekend"] = data[column_mapping["date"]].dt.dayofweek >= 5
            weekend_revenue = data.groupby("IsWeekend")[column_mapping["revenue"]].sum()
            metrics["weekend_vs_weekday_revenue"] = {
                "weekend": float(weekend_revenue[True]),
                "weekday": float(weekend_revenue[False])
            } if len(weekend_revenue) == 2 else {"weekend": 0, "weekday": 0}
        if column_mapping["customer_id"] and column_mapping["date"]:
            daily_customers = data.groupby(data[column_mapping["date"]].dt.date)[column_mapping["customer_id"]].nunique()
            metrics["table_turnover_rate"] = round(float(daily_customers.mean()), 2) if not daily_customers.empty else 0
    
    elif business_type == "service":
        if column_mapping["revenue"]:
            metrics["average_revenue_per_client"] = metrics["average_revenue"]
        if column_mapping["customer_id"]:
            metrics["total_clients"] = data[column_mapping["customer_id"]].nunique()
            repeat_clients = data[column_mapping["customer_id"]].value_counts()
            metrics["repeat_client_percent"] = round((sum(repeat_clients > 1) / metrics["total_clients"]) * 100, 2) if metrics["total_clients"] > 0 else 0
            if column_mapping["date"]:
                monthly_retention = data.groupby(data[column_mapping["date"]].dt.to_period("M"))[column_mapping["customer_id"]].nunique()
                metrics["client_retention_trend"] = round(monthly_retention.pct_change().mean() * 100, 2) if not pd.isna(monthly_retention.pct_change().mean()) else 0
        if column_mapping["category"] and column_mapping["revenue"]:
            service_revenue = data.groupby(column_mapping["category"])[column_mapping["revenue"]].sum()
            metrics["revenue_by_service"] = {k: float(v) for k, v in service_revenue.to_dict().items()}
            if column_mapping["rating"]:
                service_rating = data.groupby(column_mapping["category"])[column_mapping["rating"]].mean()
                metrics["service_rating_variance"] = round(float(service_rating.var()), 2) if len(service_rating) > 1 else 0
        if column_mapping["date"] and column_mapping["revenue"]:
            data["Day"] = data[column_mapping["date"]].dt.day_name()
            metrics["peak_service_day"] = data.groupby("Day")[column_mapping["revenue"]].sum().idxmax()
    
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
    
    try:
        business_type = getattr(uploaded_data, "business_type", "general")
        data_summary = uploaded_data.describe().to_string()
        metrics_response = await get_metrics()
        metrics = json.loads(metrics_response.body.decode('utf-8'))
        metrics_str = "\n".join([f"{k}: {v}" for k, v in metrics.items()])
        
        charts = {}
        if column_mapping["date"] and column_mapping["revenue"]:
            plt.figure(figsize=(10, 6))
            if business_type in ["cafe", "restaurant"]:
                daily_data = uploaded_data.groupby(uploaded_data[column_mapping["date"]].dt.day_name())[column_mapping["revenue"]].mean()
                daily_data.plot(kind="bar", color="orange" if business_type == "cafe" else "purple")
                plt.title(f"Average Revenue by Day ({business_type})")
                plt.xlabel("Day of Week")
            else:
                monthly_data = uploaded_data.groupby(uploaded_data[column_mapping["date"]].dt.to_period("M"))[column_mapping["revenue"]].sum()
                monthly_data.plot(kind="line", marker="o")
                plt.title("Revenue Trend by Month")
                plt.xlabel("Month")
            plt.ylabel("Revenue")
            plt.grid(True)
            plt.xticks(rotation=45)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts["revenue_trend"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close()

        # Дополнительные графики
        if business_type == "shop" and column_mapping["category"] and column_mapping["quantity"]:
            plt.figure(figsize=(10, 6))
            category_sales = uploaded_data.groupby(column_mapping["category"])[column_mapping["quantity"]].sum()
            category_sales.plot(kind="bar", color="green")
            plt.title("Sales by Category")
            plt.xlabel("Category")
            plt.ylabel("Quantity Sold")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts["sales_by_category"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close()

        if business_type == "online" and column_mapping["date"]:
            plt.figure(figsize=(10, 6))
            daily_orders = uploaded_data.groupby(uploaded_data[column_mapping["date"]].dt.date).size()
            daily_orders.plot(kind="line", marker="o", color="blue")
            plt.title("Order Volume Trend")
            plt.xlabel("Date")
            plt.ylabel("Number of Orders")
            plt.grid(True)
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts["order_trend"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close()

        if business_type == "restaurant" and column_mapping["category"] and column_mapping["revenue"]:
            plt.figure(figsize=(10, 6))
            category_data = uploaded_data.groupby(column_mapping["category"])[column_mapping["revenue"]].sum()
            category_data.plot(kind="pie", autopct='%1.1f%%')
            plt.title("Revenue by Category")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts["revenue_by_category"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close()

        if business_type == "service" and column_mapping["customer_id"]:
            plt.figure(figsize=(10, 6))
            repeat_counts = uploaded_data[column_mapping["customer_id"]].value_counts()
            repeat_counts.value_counts().plot(kind="bar", color="teal")
            plt.title("Client Visit Frequency")
            plt.xlabel("Number of Visits")
            plt.ylabel("Number of Clients")
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            charts["visit_frequency"] = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
            plt.close()

        prompt = (
            f"Analyze this {business_type} business data summary and metrics:\n\n"
            f"Data Summary:\n{data_summary}\n\n"
            f"Metrics:\n{metrics_str}\n\n"
            f"Provide detailed insights, identify anomalies, key trends, and actionable recommendations tailored for a {business_type} business."
        )
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        payload = {
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1000,
            "temperature": 0.5
        }
        response = requests.post(OPENAI_API_URL, json=payload, headers=headers)
        response.raise_for_status()
        ai_response = response.json()["choices"][0]["message"]["content"]
        
        result = {"insights": ai_response}
        result.update(charts)
        return JSONResponse(content=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error preparing data for AI: {str(e)}")

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
