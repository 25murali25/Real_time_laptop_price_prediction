import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib


filepath = r"C:\Users\HAI\OneDrive\Desktop\machine learning pratice\machine learning pratice folder\ml_pratice\df.csv"
class load_data:
    def __init__(self,filepath):
        self.filepath = filepath
        
    def load(self):
        try:
            data = pd.read_csv(self.filepath)
            df = pd.DataFrame(data)
            print('File was Sucessfully loaded')
            return df
        except Exception as e:
            print(f'Error occurred while loading the file: {e}')
            return None, None, None


class Prepocessing:
    def __init__(self,df):
        self.df = df
        
    def analysis(self):
        try:
            check_null_values = self.df.isnull().sum()
            check_numarical_values = self.df.select_dtypes(include = "number").columns
            check_cetagorical_columns = self.df.select_dtypes(include = 'object').columns 
            return self.df,check_null_values,check_numarical_values,check_cetagorical_columns        
        except Exception as e:
            print('Not Sucessfully Analysis',e)
            return None,None,None 

class featureScaling:
    def __init__(self,df):
        self.df = df
        
    def scaling(self): 
        try:
            from sklearn.preprocessing import StandardScaler,OneHotEncoder
            from sklearn.pipeline import Pipeline
            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.model_selection import train_test_split,GridSearchCV
            from sklearn.linear_model import LinearRegression
            from sklearn.tree import DecisionTreeRegressor
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.svm import SVR
            from sklearn.linear_model import Ridge,Lasso
            
            
            X = self.df.drop('Price',axis=1)
            Y = self.df['Price']
            x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)
            
            num_cols = X.select_dtypes(include = 'number').columns
            cat_cols = X.select_dtypes(include = 'object').columns
            
            num_pipe = Pipeline([
                ('imputer',SimpleImputer(strategy='median')),
                ('scaler',StandardScaler())
            ])
            
            cat_pipe = Pipeline([
                ('imputer',SimpleImputer(strategy='most_frequent')),
                ('encoder',OneHotEncoder(handle_unknown='ignore'))
            ])
            
            preprocesser = ColumnTransformer([
                ('num',num_pipe,num_cols),
                ('cat',cat_pipe,cat_cols)
            ])
            
            pipe = Pipeline([
                ("preprocessing",preprocesser),
                ("model",LinearRegression())
            ])
            
            grid = [
                {
                    'model':[LinearRegression()],
                    'model__fit_intercept':[True,False],
                },
                {
                    'model':[DecisionTreeRegressor()],
                    'model__criterion':['squared_error','friedman_mse','absolute_error','poisson'],
                    'model__random_state':[42]
                },
                {
                    "model":[RandomForestRegressor()],
                    "model__n_estimators":[100,200],
                    "model__criterion":['squared_error','friedman_mse','absolute_error','poisson'],
                    "model__random_state":[42]
                },
                {
                    "model":[SVR()],
                    "model__C":[1,10],
                    "model__kernel":['linear','rbf'],
                    "model__gamma":['scale','auto']
                },
                {
                    "model":[Ridge()],
                    "model__alpha":[0.1,1.0,10.0]
                },
                {
                    "model":[Lasso()],
                    "model__alpha":[0.1,1.0,10.0]

                }
            ] 
            
            grid_search = GridSearchCV(pipe,grid,cv=5,n_jobs=-1,verbose=2,scoring='r2')
            grid_search.fit(x_train,y_train)
            predict_y = grid_search.predict(x_test)  
            return grid_search,predict_y,y_test
        except Exception as e:
            print('Not Sucessfully Scaling',e)
            return None, None, None
        
class evalution:
    def __init__(self,grid_search,predict_y,y_test):
        self.grid_search = grid_search
        self.predict_y = predict_y
        self.y_test = y_test
        
    def model_evelation(self):
        try:
            from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
            r2 = r2_score(self.y_test,self.predict_y)
            mae = mean_absolute_error(self.y_test,self.predict_y)
            mse = mean_squared_error(self.y_test,self.predict_y)
            return r2,mae,mse
        except Exception as e:
            print('Not Sucessfully Evalution',e)  
            return None, None, None  
               


object = load_data(filepath)
loaded = object.load() 
preprocessing_loaded = Prepocessing(loaded)
df,check_null_values,check_numarical_values,check_cetagorical_columns  = preprocessing_loaded.analysis()
feature_scaling = featureScaling(df)
grid_search,predict_y,y_test = feature_scaling.scaling()
evalution_model_mextrics = evalution(grid_search,predict_y,y_test)
result = evalution_model_mextrics.model_evelation()

joblib.dump(grid_search.best_estimator_,"Model.pkl")


from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
app = FastAPI()


@app.on_event("startup")
def load_model():
    global input_model
    input_model = joblib.load("Model.pkl")

class InputData(BaseModel):
    def name_def():
        Company:str
        TypeName:str
        Ram:float
        Weight:float
        Touchscreen:int
        Ips:int
        ppi:float
        Cpu_brand:str
        HDD:int
        SSD:int
        Gpu_brand:str
        os:str
        
        
@app.post('/predict')
def model_predict(data:InputData):
    try:
        input_data = [[data.Company,data.TypeName,data.Ram,data.Weight,data.Touchscreen,data.Ips,data.ppi,data.Cpu_brand,data.HDD,data.SSD,data.Gpu_brand,data.os]]
        prediction = input_model.predict(input_data)
        return {"predicted_price": prediction[0]}
    except Exception as e:
        return {"error": str(e)}
    
if __name__ == "__main__":
    uvicorn.run(app,port=8000)           
        
        

    
