# AI Lens — Plant Leaf Disease Detection & Agronomic Advisory Web Application

**AI Lens** is an end-to-end deep learning and computer vision web application designed to identify plant leaf diseases across **14 agricultural crop species** and **38 distinct condition classes** using transfer learning with **MobileNetV2**.

Developed as a college engineering capstone project, AI Lens combines an optimized convolutional neural network with an intuitive, presentation-grade Streamlit web interface. It delivers instantaneous botanical diagnoses, model confidence estimates, visual symptom analyses, and curated cultural and chemical treatment recommendations.

---

## 🌟 Key Features

- **Transfer Learning with MobileNetV2**: Pretrained on ImageNet, fine-tuned on leaf textures, offering lightweight, rapid inference suitable for CPU edge deployment and GPU training.
- **38 Disease & Healthy Classes**: Covers major agricultural staples including Tomato, Potato, Apple, Corn, Grape, Bell Pepper, Peach, Strawberry, and Cherry.
- **Comprehensive Botanical Knowledge Base**: Built-in agronomic guidance (`disease_info.py`) detailing causal pathogens, observable symptoms, cultural sanitation practices, preventative measures, and chemical controls.
- **Out-of-Distribution & Uncertainty Awareness**: Flags low-confidence predictions, ambiguous inter-class margins, and potential non-leaf inputs.
- **Presentation-Grade Streamlit Interface**: Clean, responsive layout with leaf-themed styling, interactive confidence thresholds, top-3 class distributions, and visual status cards.
- **Graceful Weight Handling**: Runs in an educational simulation mode when weights are not yet present, avoiding runtime crashes during reviews or demonstrations.
- **Reproducible ML Pipeline**: Fully separate train/val/test splits, training-only data augmentation, early stopping, and automated evaluation reports (Macro F1, Precision, Recall, Confusion Matrix).

---

## 📂 Project Architecture

```text
AI-Lens/
├── app.py                      # Main Streamlit web application
├── disease_info.py             # Botanical metadata & care recommendations for 38 classes
├── utils.py                    # Preprocessing, inference, and uncertainty detection helpers
├── train.py                    # Complete TensorFlow/Keras training & evaluation pipeline
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── .gitignore                  # Git ignore rules
├── models/
│   ├── .gitkeep
│   ├── class_names.json        # Canonical list of 38 PlantVillage classes
│   └── plant_disease_mobilenetv2.keras (generated upon training)
├── data/
│   └── .gitkeep                # Local dataset storage (ignored by Git)
├── reports/
│   ├── .gitkeep
│   ├── confusion_matrix.png    # Evaluation heatmap (generated upon evaluation)
│   ├── training_history.png    # Loss & accuracy curves (generated upon training)
│   ├── classification_report.txt
│   └── metrics.json
└── notebooks/
    └── training.ipynb          # Google Colab / Kaggle GPU training notebook
```

---

## 📊 Dataset: PlantVillage

AI Lens utilizes the official **PlantVillage** dataset, containing 54,303 expert-curated photographs of healthy and diseased crop leaves across 38 class categories.

- **Hugging Face Hub**: [`mohanty/PlantVillage`](https://huggingface.co/datasets/mohanty/PlantVillage) (Color configuration)
- **Crops Monitored (14 species)**: Apple, Blueberry, Cherry, Corn (Maize), Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, and Tomato.

### Dataset Citation
```bibtex
@article{mohanty2016using,
  title={Using deep learning for image-based plant disease detection},
  author={Mohanty, Sharada P and Hughes, David P and Salath{\'e}, Marcel},
  journal={Frontiers in plant science},
  volume={7},
  pages={1419},
  year={2016},
  publisher={Frontiers}
}
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/AI-Lens.git
cd AI-Lens
```

### 2. Create a Python Virtual Environment
We recommend Python 3.10, 3.11, or 3.12:
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Web Application Locally

To start the Streamlit web server:

```bash
streamlit run app.py
```

The web interface will automatically open in your default browser at `http://localhost:8501`.

> **Note on Model Weights**:
> If you have not yet trained the model, `app.py` detects the absence of weights in `models/` and starts in **Educational Demo & Simulation Mode**. You can immediately test the leaf upload interface, inspect diagnoses, and explore agronomic recommendations without any runtime errors.

---

## 🧠 Model Training & Evaluation

You have two convenient options to train the MobileNetV2 classifier:

### Option A: Free GPU Training on Google Colab (Recommended)
1. Upload [`notebooks/training.ipynb`](notebooks/training.ipynb) to [Google Colab](https://colab.research.google.com/).
2. Select **Runtime** &rarr; **Change runtime type** &rarr; **T4 GPU**.
3. Run all cells sequentially. The notebook programmatically streams the dataset from Hugging Face, applies data augmentation, trains the MobileNetV2 backbone, generates confusion matrices, and triggers an automated download of `plant_disease_mobilenetv2.keras`.
4. Place the downloaded file into your local `AI-Lens/models/` folder and refresh your browser.

### Option B: Local Training via CLI
You can train directly using `train.py`:

```bash
# Train using direct download of official PlantVillage archive from Hugging Face:
python train.py --use_hf --epochs 15 --batch_size 32

# Or train using an existing extracted dataset directory:
python train.py --data_dir data/plantvillage_color --epochs 20 --batch_size 32 --fine_tune
```

### Training Pipeline Features:
- **Leaf-Level Disjoint Splitting**: Preserves 70% Train, 15% Validation, 15% Test partitions to prevent data leakage.
- **Augmentation Pipeline**: Applied strictly to training data (RandomFlip, RandomRotation, RandomZoom, RandomContrast).
- **Callbacks**: `EarlyStopping` (patience=5), `ModelCheckpoint` (monitors `val_accuracy`), and `ReduceLROnPlateau`.
- **Automated Outputs**: Saves evaluation plots and metrics into `reports/`:
  - `reports/training_history.png`
  - `reports/confusion_matrix.png`
  - `reports/classification_report.txt`
  - `reports/metrics.json`

---

## 🌐 Cloud Deployment Guide

### Deploying to Streamlit Community Cloud
1. Push your repository to GitHub (ensure `models/plant_disease_mobilenetv2.keras` is tracked via Git LFS or hosted on Hugging Face Model Hub if over 100MB; MobileNetV2 `.keras` is ~14MB and can be pushed directly).
2. Visit [share.streamlit.io](https://share.streamlit.io) and log in with GitHub.
3. Select your repository, specify branch `main`, and set the entry point to `app.py`.
4. Click **Deploy**.

### Deploying with Docker
A standard `Dockerfile` can be configured as follows:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## ⚠️ Scientific & Practical Limitations

1. **Not a Clinical Agronomic Diagnosis**: AI Lens is an algorithmic screening tool designed for academic research and educational decision-support. It is not an accredited substitute for professional agricultural laboratory tests or in-person pathology examinations.
2. **Softmax Confidence Calibration**: Softmax probabilities indicate relative classification confidence across the 38 closed-set categories and should not be construed as calibrated statistical certainty.
3. **Out-of-Distribution Inputs**: Photographs of non-plant subjects, weeds, soil, or crops outside the 14 trained species will produce unreliable predictions. Always observe the on-screen uncertainty indicators.
4. **Lighting & Angle Dependency**: The PlantVillage dataset features controlled lab backgrounds. Performance on complex field backgrounds with overlapping leaves or severe shadows may exhibit reduced accuracy compared to laboratory benchmarks.

---

## 🔮 Future Enhancements

- **In-Field Object Detection (YOLOv8 / YOLOv10)**: Real-time multi-leaf localization and bounding-box segmentation directly in complex orchard environments.
- **Multilingual Support**: Providing advisory and care recommendations in regional farming languages (Hindi, Spanish, Swahili, etc.).
- **Weather API Integration**: Incorporating real-time local humidity and rainfall data to calculate infection risk scores dynamically.
- **Lightweight TFLite & ONNX Quantization**: Enabling sub-10ms offline inference directly on low-cost edge microcontrollers and mobile devices.

---

## 📜 License
This project is licensed under the MIT License for educational and academic demonstration purposes.
