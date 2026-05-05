# FruitVision AI UI

Premium Next.js frontend for the Fruit Quality Detector for SMS project.

## How It Works

The browser sends an uploaded image to `POST /api/detect`.

- `DETECTION_MODE=mock`: returns a realistic sample result after 1.5 seconds.
- `DETECTION_MODE=backend`: forwards the image to the existing FastAPI backend at `http://127.0.0.1:8000/detect` and normalizes the output.

The Python backend is not changed by this frontend.

## Run

```powershell
cd C:\Users\moham\OneDrive\Documents\FruitQualityWeb_Rebuild\fruitvision-ui
npm install
npm run dev
```

Open:

```text
http://127.0.0.1:3000
```

## Backend Mode

Start the existing FastAPI backend first:

```powershell
cd C:\Users\moham\OneDrive\Documents\FruitQualityWeb_Rebuild
.\run_website.bat
```

Then in another terminal:

```powershell
cd C:\Users\moham\OneDrive\Documents\FruitQualityWeb_Rebuild\fruitvision-ui
$env:DETECTION_MODE="backend"
$env:FASTAPI_DETECT_URL="http://127.0.0.1:8000/detect"
npm run dev
```

## Future Model Integration

Keep the frontend contract stable:

```json
{
  "fruit_name": "Apple",
  "quality": "Fresh",
  "grade": "A",
  "freshness_score": 92,
  "confidence": 96.4,
  "defects": ["No major defects detected"],
  "recommendation": "Safe to buy"
}
```

YOLO, CNN, TensorFlow/Keras, ConvNeXt, EfficientNet, or TFLite models can be connected behind `/api/detect` without changing the UI.
