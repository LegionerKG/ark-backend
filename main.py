from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends, status, Response
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.middleware.cors import CORSMiddleware
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import io
import base64
from openai import OpenAI
import os
from dotenv import load_dotenv
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pytrends.request import TrendReq
import tempfile
from typing import Optional
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import logging

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("ark_backend.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()

app = FastAPI(
    title="ARK Backend",
    description="API для анализа бизнес-данных с авторизацией",
    version="1.0.0"
)

# Добавляем CORS для работы с frontend
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","https://ark-frontend.onrender.com"],  # Разрешаем фронтенд-домен
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Настройки JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Хэширование паролей
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

# Настройка базы данных SQLite
DATABASE_URL = "sqlite:///./users.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Модель пользователя в базе данных
class UserDB(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)

Base.metadata.create_all(bind=engine)

# Зависимость для получения сессии БД
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Модели для авторизации
class User(BaseModel):
    username: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

# Вспомогательные функции
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def authenticate_user(username: str, password: str, db: Session):
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if not user or not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    if not token:
        raise credentials_exception
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserDB).filter(UserDB.username == username).first()
    if user is None:
        raise credentials_exception
    return user

# OpenAI и Google Trends
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API key not configured")
    raise HTTPException(status_code=500, detail="OpenAI API key not configured")
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
pytrends = TrendReq(hl='en-US', tz=360)

@app.get("/")
async def root():
    logger.info("Root endpoint accessed")
    return {"message": "Welcome to ARK Backend API"}

# Регистрация
@app.post("/register")
async def register(user: User, db: Session = Depends(get_db)):
    logger.info(f"Register attempt for username: {user.username}")
    db_user = db.query(UserDB).filter(UserDB.username == user.username).first()
    if db_user:
        logger.warning(f"Username {user.username} already registered")
        raise HTTPException(status_code=400, detail="Username already registered")
    hashed_password = get_password_hash(user.password)
    new_user = UserDB(username=user.username, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"User {user.username} registered successfully")
    return {"message": "User registered successfully"}

# Вход с установкой HttpOnly cookie
@app.post("/token")
async def login_for_access_token(response: Response, form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    logger.info(f"Login attempt for username: {form_data.username}")
    user = authenticate_user(form_data.username, form_data.password, db)
    if not user:
        logger.warning(f"Failed login attempt for username: {form_data.username}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user.username}, expires_delta=access_token_expires
    )
    response.set_cookie(
        key="access_token",
        value=access_token,
        httponly=True,
        secure=True,  # Установите False для localhost, True для продакшена
        samesite="lax",
        max_age=ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )
    logger.info(f"User {form_data.username} logged in successfully")
    return {"message": "Login successful"}

# Выход
@app.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    logger.info("User logged out")
    return {"message": "Logout successful"}

# Загрузка файла
@app.post("/upload")
async def upload_file(file: UploadFile = File(...), current_user: UserDB = Depends(get_current_user)):
    logger.info(f"User {current_user.username} uploading file: {file.filename}")
    try:
        if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            logger.error(f"Invalid file format: {file.filename}")
            raise HTTPException(status_code=400, detail="Only CSV or Excel files are supported")

        content = await file.read()
        if file.filename.endswith('.csv'):
            df = pd.read_csv(io.BytesIO(content))
        else:
            df = pd.read_excel(io.BytesIO(content))

        if len(df.columns) < 3:
            logger.error(f"File {file.filename} has fewer than 3 columns")
            raise HTTPException(status_code=400, detail="File must have at least 3 columns")

        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        logger.info(f"File {file.filename} uploaded successfully, token: {tmp_path}")
        return JSONResponse({
            "columns": list(df.columns),
            "file_token": tmp_path,
            "message": "Please specify date, revenue, and expenses columns, or set 'auto' to true"
        })

    except Exception as e:
        logger.error(f"Error processing file {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

# Анализ с улучшенной автоинтерпретацией
@app.post("/analyze")
async def analyze_file(
    file_token: str = Form(...),
    date_col: Optional[str] = Form(None),
    revenue_col: Optional[str] = Form(None),
    expenses_col: Optional[str] = Form(None),
    auto: bool = Form(False),
    current_user: UserDB = Depends(get_current_user)
):
    logger.info(f"User {current_user.username} analyzing file with token: {file_token}")
    try:
        if not os.path.exists(file_token):
            logger.error(f"File token invalid or expired: {file_token}")
            raise HTTPException(status_code=400, detail="File token invalid or expired")
        
        if file_token.endswith('.csv'):
            df = pd.read_csv(file_token)
        else:
            df = pd.read_excel(file_token)

        if auto:
            # Улучшенная автоинтерпретация
            date_candidates = [
                col for col in df.columns
                if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.8
                or any(keyword in col.lower() for keyword in ['date', 'time', 'day', 'month', 'year'])
            ]
            if not date_candidates:
                logger.error("No date column detected")
                raise HTTPException(status_code=400, detail="Could not detect date column")
            date_col = date_candidates[0]
            logger.info(f"Auto-detected date column: {date_col}")

            numeric_cols = [
                col for col in df.columns if col != date_col
                and pd.to_numeric(df[col], errors='coerce').notna().sum() > len(df) * 0.8
            ]
            if len(numeric_cols) < 2:
                logger.error("Not enough numeric columns detected")
                raise HTTPException(status_code=400, detail="Could not detect at least two numeric columns")

            potential_revenue = []
            potential_expenses = []
            for col in numeric_cols:
                col_values = pd.to_numeric(df[col], errors='coerce').dropna()
                mean_value = col_values.mean()
                if any(keyword in col.lower() for keyword in ['revenue', 'sales', 'income', 'profit']):
                    potential_revenue.append((col, mean_value))
                elif any(keyword in col.lower() for keyword in ['expense', 'cost', 'spend', 'loss']):
                    potential_expenses.append((col, mean_value))
                elif mean_value > 0:
                    potential_revenue.append((col, mean_value))
                else:
                    potential_expenses.append((col, mean_value))

            if not potential_revenue or not potential_expenses:
                sorted_cols = sorted(
                    [(col, pd.to_numeric(df[col], errors='coerce').mean()) for col in numeric_cols],
                    key=lambda x: x[1],
                    reverse=True
                )
                revenue_col = sorted_cols[0][0]
                expenses_col = sorted_cols[1][0]
            else:
                revenue_col = max(potential_revenue, key=lambda x: x[1])[0]
                expenses_col = max(potential_expenses, key=lambda x: x[1])[0]

            logger.info(f"Auto-detected revenue: {revenue_col}, expenses: {expenses_col}")
        else:
            if not (date_col and revenue_col and expenses_col):
                logger.error("Manual column selection incomplete")
                raise HTTPException(status_code=400, detail="Please provide date_col, revenue_col, and expenses_col unless auto is true")
            if not all(col in df.columns for col in [date_col, revenue_col, expenses_col]):
                logger.error(f"Specified columns not found: {date_col}, {revenue_col}, {expenses_col}")
                raise HTTPException(status_code=400, detail="Specified columns not found in file")

        df = df.rename(columns={date_col: 'date', revenue_col: 'revenue', expenses_col: 'expenses'})
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['expenses'] = pd.to_numeric(df['expenses'], errors='coerce')
        df = df.dropna(subset=['date', 'revenue', 'expenses'])

        if df.empty:
            logger.error("No valid data after interpretation")
            raise HTTPException(status_code=400, detail="No valid data after interpretation")

        total_revenue = df['revenue'].sum()
        total_expenses = df['expenses'].sum()
        profit = total_revenue - total_expenses
        avg_revenue = df['revenue'].mean()
        avg_expenses = df['expenses'].mean()
        breakeven_point = total_expenses / (avg_revenue - avg_expenses) if (avg_revenue - avg_expenses) != 0 else float('inf')
        profit_margin = (profit / total_revenue) * 100 if total_revenue != 0 else 0

        df['month'] = df['date'].dt.to_period('M')
        monthly_trends = df.groupby('month').agg({'revenue': 'sum', 'expenses': 'sum'}).reset_index()
        monthly_trends['profit'] = monthly_trends['revenue'] - monthly_trends['expenses']

        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df['revenue'], label='Revenue', color='green')
        plt.plot(df['date'], df['expenses'], label='Expenses', color='red')
        plt.legend()
        plt.title("Revenue and Expenses Over Time")
        plt.xticks(rotation=45)
        buffer = io.BytesIO()
        plt.savefig(buffer, format="png", bbox_inches="tight")
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        plt.close()

        df['date_ordinal'] = df['date'].map(lambda x: x.toordinal())
        X = df['date_ordinal'].values.reshape(-1, 1)
        y = df['revenue'].values
        model = LinearRegression().fit(X, y)
        last_date = df['date'].iloc[-1]
        future_dates = pd.date_range(last_date, periods=30, freq='D')
        X_future = [[d.toordinal()] for d in future_dates]
        forecast = model.predict(X_future).tolist()

        prompt = (
            f"Business data: Total Revenue: {total_revenue}, Total Expenses: {total_expenses}, "
            f"Profit: {profit}, Profit Margin: {profit_margin:.2f}%, Breakeven Point: {breakeven_point:.2f}. "
            f"Monthly Trends: {monthly_trends.to_dict('records')}. Provide actionable business advice."
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=500
        )

        ai_advice = response.choices[0].message.content.strip()

        pytrends.build_payload(kw_list=['sales'], timeframe='today 3-m', geo='US')
        trends_data = pytrends.interest_over_time()
        market_trend = trends_data['sales'].mean() if not trends_data.empty else 0

        os.remove(file_token)
        logger.info(f"Analysis completed for user {current_user.username}")

        return JSONResponse({
            "metrics": {
                "total_revenue": total_revenue,
                "total_expenses": total_expenses,
                "profit": profit,
                "profit_margin": profit_margin,
                "breakeven_point": breakeven_point
            },
            "trends": monthly_trends.to_dict('records'),
            "chart": f"data:image/png;base64,{img_base64}",
            "forecast": {
                "dates": [d.strftime('%Y-%m-%d') for d in future_dates],
                "values": forecast
            },
            "ai_advice": ai_advice,
            "market_trend": float(market_trend),
            "interpreted_columns": {"date": date_col, "revenue": revenue_col, "expenses": expenses_col}
        })

    except Exception as e:
        logger.error(f"Error analyzing file {file_token}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

# Экспорт PDF
@app.post("/export-pdf")
async def export_pdf(file_token: str = Form(...), current_user: UserDB = Depends(get_current_user)):
    logger.info(f"User {current_user.username} exporting PDF for file token: {file_token}")
    try:
        if not os.path.exists(file_token):
            logger.error(f"File token invalid or expired: {file_token}")
            raise HTTPException(status_code=400, detail="File token invalid or expired")
        
        if file_token.endswith('.csv'):
            df = pd.read_csv(file_token)
        else:
            df = pd.read_excel(file_token)

        date_col = next(col for col in df.columns if pd.to_datetime(df[col], errors='coerce').notna().sum() > len(df) * 0.8)
        numeric_cols = [col for col in df.columns if col != date_col and pd.to_numeric(df[col], errors='coerce').notna().sum() > len(df) * 0.8]
        means = {col: pd.to_numeric(df[col], errors='coerce').mean() for col in numeric_cols}
        sorted_cols = sorted(means.items(), key=lambda x: x[1], reverse=True)
        revenue_col, expenses_col = sorted_cols[0][0], sorted_cols[1][0]

        df = df.rename(columns={date_col: 'date', revenue_col: 'revenue', expenses_col: 'expenses'})
        df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
        df['expenses'] = pd.to_numeric(df['expenses'], errors='coerce')

        total_revenue = df['revenue'].sum()
        total_expenses = df['expenses'].sum()
        profit = total_revenue - total_expenses

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            c = canvas.Canvas(tmp.name, pagesize=letter)
            c.drawString(100, 750, "ARK Business Report")
            c.drawString(100, 730, f"Total Revenue: {total_revenue}")
            c.drawString(100, 710, f"Total Expenses: {total_expenses}")
            c.drawString(100, 690, f"Profit: {profit}")
            c.save()
        
        os.remove(file_token)
        logger.info(f"PDF exported successfully for user {current_user.username}")
        return FileResponse(tmp.name, filename="ark_report.pdf", media_type="application/pdf")

    except Exception as e:
        logger.error(f"Error generating PDF for {file_token}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating PDF: {str(e)}")

@app.get("/health")
async def health_check():
    logger.info("Health check accessed")
    return {"status": "healthy"}
