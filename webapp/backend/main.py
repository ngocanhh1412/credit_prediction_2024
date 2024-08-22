from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from form import app as form_app
from table import app as table_app
from predict_by_id import app as predict_by_id_app

app = FastAPI()

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1"  # React app chạy trên cổng 3000
]

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/threshold")
async def get_threshold():
    with open('threshold.txt', 'r') as file:
        threshold = file.read().strip()
    if threshold is None:
        raise HTTPException(status_code=500, detail="Error reading threshold")
    return {"threshold": threshold}

# Include the form_app routes
app.mount("/form", form_app)
app.mount("/table", table_app)
app.mount("/predict_by_id", predict_by_id_app)

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
