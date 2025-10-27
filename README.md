# XGBoost Traffic Prediction API

A FastAPI-based REST API for predicting traffic congestion using XGBoost model. The API provides real-time traffic predictions for various intersections in the city.

## Features

- Real-time traffic congestion prediction
- Support for multiple intersection points
- Congestion level and average speed estimation
- RESTful API with FastAPI
- Swagger documentation included

## Project Structure

```
xgboost-api/
├── data/
│   ├── preprocessing.py      # Data preprocessing utilities
│   └── xgboost_model.pkl    # Trained XGBoost model (Git LFS)
├── main.py                  # FastAPI application
├── requirements.txt         # Project dependencies
└── README.md               # Project documentation
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/PrajwalShetty-114/XG-Boost-Model.git
cd XG-Boost-Model
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate  # CMD
# OR
source .venv/Scripts/activate  # Git Bash
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the API server:
```bash
python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

2. Access the API:
- API Root: http://127.0.0.1:8000
- Interactive API Documentation: http://127.0.0.1:8000/docs

3. Make predictions using the API:
```bash
curl -X POST "http://127.0.0.1:8000/predict/" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "xgboost",
       "junctionName": "Intersection_Trinity Circle",
       "predictionTime": "Next Hour"
     }'
```

## API Endpoints

### GET /
- Returns API status
- Response: `{"message": "XGBoost Traffic Prediction API is running!"}`

### POST /predict/
- Makes traffic predictions for a given intersection
- Request Body:
  ```json
  {
    "model": "xgboost",
    "junctionName": "Intersection_Trinity Circle",
    "predictionTime": "Next Hour"
  }
  ```
- Response:
  ```json
  {
    "predictions": {
      "congestion": {
        "level": 0.5,
        "label": "Medium"
      },
      "avgSpeed": 30
    },
    "alternativeRoute": null
  }
  ```

## Supported Intersections

The model supports predictions for the following intersections:
- Intersection_100 Feet Road
- Intersection_Anil Kumble Circle
- Intersection_Ballari Road
- Intersection_CMH Road
- Intersection_Hebbal Flyover
- Intersection_Hosur Road
- Intersection_ITPL Main Road
- Intersection_Jayanagar 4th Block
- Intersection_Marathahalli Bridge
- Intersection_Sarjapur Road
- Intersection_Silk Board Junction
- Intersection_Sony World Junction
- Intersection_South End Circle
- Intersection_Trinity Circle
- Intersection_Tumkur Road
- Intersection_Yeshwanthpur Circle

## Dependencies

- fastapi>=0.110.0
- uvicorn[standard]>=0.23.0
- scikit-learn>=1.3.0
- pandas>=2.1.0
- xgboost>=2.0.0
- joblib>=1.3.0
- pydantic>=2.4.0
- python-multipart>=0.0.6

## Note

The XGBoost model file (`data/xgboost_model.pkl`) is stored using Git LFS due to its size. Make sure you have Git LFS installed when cloning the repository to get the actual model file.

## License

This project is licensed under the MIT License - see the LICENSE file for details.