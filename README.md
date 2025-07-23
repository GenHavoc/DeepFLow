# Major-Assignment-1

Phase 1- Used Machine Appendix to use other features and analysed the features.

Phase 2- I analysed the data, the datatypes and used a threshold of 80% to drop a column. Used simple imputer for missing values, initally used KNN but the computation was very slow.

Imputed both train and test categories with missing value. Used ordinal encoder and one hot encoding for categories.

Dropped columns with more than 60% data missing and aligned test and train columns. If a column is present in train and not present in test, I imputed with median of train data.

Phase 3- Applied XGBoost along with optuna (read the documentation ) for tuning hyperparamters and Random forest and compared their result with R2 and RMSE. Applied ridge regression with normalisation but got inferior results.

Phase 4- Used test file for predictions.

# Major Assignment-2

Used EfficientNetB3 and MobileNetV3 as pre trained models. MobileNet due to its light architecture gave faster results on all base layers freezed than EfficientNetV3.
Transfer Learning-
1)  Pretrained on ImageNet as frozen base models and trained a custom classifier on top.
2) Unfroze the base model and continued training with a lower learning rate to adapt deeper layers to the dog breed classification task.

## 📊 Performance Metrics

| Metric              | Value       |
|---------------------|-------------|
| Training Accuracy   | ~95%        |
| Validation Accuracy | ~80%        |
| Loss Function       | Categorical Crossentropy |
| Optimizer           | Adam        |

## 📦 Files

| File              | Description                                |
|-------------------|--------------------------------------------|
| `cnn_model.h5`    | Final trained Keras model (HDF5 format)    |
| `submission.csv`  | Predictions on test dataset (id, label)    |
| `notebook.ipynb`  | Training and preprocessing code            |

---
## Load the Saved Model

```python
from tensorflow.keras.models import load_model

model = load_model("cnn_model.h5")


import numpy as np
img = ...  # load the image here 
prediction = model.predict(np.expand_dims(img, axis=0))
```
------

# FastAPI CNN Image Classification API

This FastAPI app serves a CNN model for dog breed classification.

## 📁 Requirements

- `main.py`
- `model.pth` (your trained CNN model)
- `label_encoder.pkl` (for decoding model output to labels)

All three files must be in the **same directory**, e.g., `Deepflow/`.

---

## Running the API

1. **Open terminal** and change to the directory containing the files:

   ```bash
   cd deepflow
   ```

2. **Start the FastAPI server**:

   ```bash
   uvicorn main:app --reload
   ```

3. **Open your browser** and visit:

   ```
   http://127.0.0.1:8000/docs
   ```

   You’ll see an interactive Swagger UI to test the API.

---

## 📦 API Endpoint

### `POST /predict/`

- **Input:** an image file (e.g., `.jpg`, `.png`)
- **Response:** JSON with predicted class label

#### Example using Swagger UI:

1. Click `Try it out`
2. Upload an image file
3. Click `Execute`
4. Get back the predicted label 

---

## 📝 Example Response

```json
{
  "prediction": "Great Dane"
}
```

---







