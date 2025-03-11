# API Documentation

# Spectrum Prediction API Documentation

This API is built with Flask and provides endpoints to predict the class of a spectrum based on provided data. It leverages a pre-trained model and a feature extraction module to compute predictions from spectrum energy and intensity values.

API URL: [https://spectrum-classifier.fly.dev/api](https://spectrum-classifier.fly.dev/api)

---

## 1. GET `https://spectrum-classifier.fly.dev/api`

#### Description
A simple endpoint that returns a greeting message.

#### Request

- **Method:** GET  
- **URL:** `/api`

#### Response

- **Status Code:** 200 OK  
- **Content-Type:** `application/json`  
- **Body Example:**
  ```json
  {
    "message": "Hello, world!"
  }
  ```

---

## 2. POST `https://spectrum-classifier.fly.dev/api/predict`

#### Description
Accepts spectrum data and returns a prediction along with additional feature details. Data can be provided as a JSON payload containing `energy` and `intensity` arrays.

> **Note:** Although there is a helper function for file processing (`predict_spectrum_from_file`), the `/api/predict` endpoint expects JSON input.

#### Request

- **Method:** POST  
- **URL:** `/api/predict`  
- **Content-Type:** `application/json`

##### JSON Payload Format

- **Required Keys:**
  - `energy`: An array of energy values.
  - `intensity`: An array of intensity values.
  
- **Example:**
  ```json
  {
    "energy": [1.0, 2.0, 3.0, 4.0],
    "intensity": [10.0, 20.0, 30.0, 40.0]
  }
  ```

- **Validation:**
  - Both `energy` and `intensity` arrays must be present.
  - Arrays must have the same shape.

#### Processing Steps

1. **Validation:**  
   The API verifies the presence of both `energy` and `intensity` arrays in the JSON payload and confirms that they are of matching shape.
   
2. **Feature Extraction:**  
   Converts the arrays into NumPy arrays and uses the `extract_features` function to extract features from the spectrum data.
   
3. **Prediction:**  
   Reshapes the extracted features to match the model's expected 2D array format and performs a prediction using the pre-trained model.
   
4. **Response Construction:**  
   If the extracted features are valid (i.e., not all zeros), the API returns:
   - The predicted class.
   - Additional feature details:
     - `main_peak_energy`
     - `peak_centers`
     - `peak_amplitudes`

#### Response

- **Success (200 OK):**
  ```json
  {
    "prediction": 1.0,
    "main_peak_energy": 2.5,
    "peak_centers": [0.1, 0.2, 0.3],
    "peak_amplitudes": [5.0, 10.0, 15.0]
  }
  ```

- **Error Responses:**

  - **Missing Keys:**  
    If the JSON does not contain both `energy` and `intensity`:
    ```json
    {
      "error": "JSON must contain \"energy\" and \"intensity\" arrays"
    }
    ```
    **Status Code:** 400

  - **Array Shape Mismatch:**  
    If the provided `energy` and `intensity` arrays do not match in shape:
    ```json
    {
      "error": "Energy and intensity arrays must have the same shape"
    }
    ```
    **Status Code:** 400

  - **Invalid Features:**  
    If the feature extraction yields all zeros:
    ```json
    {
      "error": "Extracted features are all zeros"
    }
    ```
    **Status Code:** 400

---
