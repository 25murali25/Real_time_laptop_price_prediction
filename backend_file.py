from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import uvicorn
app = FastAPI()


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app):
    global input_model
    with open('best_model.pkl', 'rb') as f:
        input_model = pickle.load(f)
    yield

app = FastAPI(lifespan=lifespan)

from pydantic import BaseModel, Field

class InputData(BaseModel):
    Company: str = Field(alias="company")
    TypeName: str
    Ram: float
    Weight: float
    Touchscreen: int
    Ips: int
    ppi: float
    Cpu_brand: str
    HDD: int
    SSD: int
    Gpu_brand: str
    os: str

    class Config:
        populate_by_name = True
        
        
@app.post('/predict')
def model_predict(data: InputData):
    try:
        import pandas as pd
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data.model_dump()])

        # Ensure correct column order
        expected_cols = ['Company', 'TypeName', 'Ram', 'Weight',
                         'Touchscreen', 'Ips', 'ppi', 'Cpu_brand',
                         'HDD', 'SSD', 'Gpu_brand', 'os']

        input_df = input_df.reindex(columns=expected_cols)

        # Prediction
        prediction = input_model.predict(input_df)

        return {"predicted_price": float(prediction[0])}

    except Exception as e:
        return {"error": str(e)}
    
    
if __name__ == "__main__":
    uvicorn.run(app,host="0.0.0.0",port=8005) 