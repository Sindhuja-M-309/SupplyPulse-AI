# SupplyPulse AI: Multimodal Edge Logistics Control Tower

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8)
![Librosa](https://img.shields.io/badge/Librosa-Signal%20Processing-orange)
![SQLite](https://img.shields.io/badge/SQLite-Audit%20Persistence-003B57)

**SupplyPulse AI** is a zero-hardware, real-time perishable supply chain intelligence platform. By fusing acoustic resonance analysis with explainable computer vision (XAI) on standard mobile devices, the system detects internal and external crop degradation during transit, automates dynamic geospatial rerouting, and mitigates post-harvest financial losses through secondary market liquidation.

---

## Executive Summary

Post-harvest fresh produce losses in India exceed $11 billion annually, with an estimated 30–40% of spoilage occurring in transit due to ambient heat spikes, highway delays, and lack of real-time cargo telemetry. Existing cold-chain IoT hardware imposes prohibitive capital expenditures on smallholder farmers and regional logistics operators, while visual surface inspection alone cannot detect internal cellular breakdown prior to irreversible spoilage.

SupplyPulse AI eliminates hardware barriers by leveraging smartphone sensors as edge diagnostic tools, coupling predictive shelf-life modeling with automated B2B revenue-rescue workflows.

---

## System Architecture & Capabilities

1. **Multimodal Edge Diagnostics**
   - **Acoustic Resonance Engine (`Librosa`, `Scikit-Learn`):** Extracts Mel-Frequency Cepstral Coefficients (MFCCs) and spectral centroids from produce tap audio recordings to quantify internal turgor pressure and firmness without invasive testing.
   - **Explainable Computer Vision (`OpenCV`):** Evaluates surface imagery of cargo batches to identify early-stage bruising and fungal rot, rendering bounding-box defect overlays for visual auditability.

2. **Predictive Risk & Telemetry Integration**
   - **Decay Risk Index:** Combines multimodal sensor outputs with elapsed transit duration and live meteorological data via the **Open-Meteo REST API** to compute remaining shelf life in hours.
   - **Cold-Chain Visualization (`Plotly`):** Renders real-time environmental and degradation trend curves across active transit routes.

3. **Autonomous Decision & Revenue Protection**
   - **Dynamic GIS Rerouting (`Folium`):** Automatically recalculates and updates transit polylines to the nearest regional wholesale market (*mandi*) when the Decay Risk Index exceeds the 45% critical threshold.
   - **B2B Distress Liquidation Engine:** Lists at-risk batches on a secondary buyer board (connecting with regional juice, pulp, and sauce processors) at calculated discounts to recover 60–70% of cargo value rather than incurring a total write-off.
   - **Multilingual Voice Dispatch (`gTTS`):** Synthesizes hands-free spoken reroute directives in Tamil and English for vehicle operators.
   - **Compliance & Audit Logging (`ReportLab`, `SQLite`):** Generates downloadable PDF Transit Quality Certificates and maintains immutable evaluation logs in a relational database.

---

## Technology Stack

| Component | Technologies |
| :--- | :--- |
| **Application & Visualization** | Python, Streamlit, Plotly |
| **Acoustic Signal Processing** | Librosa, SciPy, Scikit-Learn, Joblib |
| **Computer Vision & Image Processing** | OpenCV (`cv2`), NumPy, Pillow |
| **Geospatial & External APIs** | Folium, `streamlit-folium`, Open-Meteo API |
| **Audio Synthesis, Reporting & Database** | Google Text-to-Speech (`gTTS`), ReportLab, SQLite3, Pandas |

---

## Repository Structure

```text
supplypulse-ai/
├── app.py                     # Main Streamlit Control Tower and decision engine
├── audio_engine.py            # Acoustic feature extraction and classifier training pipeline
├── vision_engine.py           # OpenCV defect detection and XAI bounding-box renderer
├── audio_firmness_model.pkl   # Serialized Scikit-Learn acoustic classification model
├── supplypulse.db             # SQLite database for transit evaluation audit logs
├── requirements.txt           # Project dependencies
└── README.md                  # System documentation
```

---

## Installation & Deployment

### Prerequisites
- Python 3.10 or higher
- Git

### Setup Instructions

1. **Clone the repository**
   ```
   git clone https://github.com/Sindhuja-M-309/supplypulse-ai.git
   cd supplypulse-ai
   ```

2. **Initialize a virtual environment**
   ```
   python -m venv venv

   # Windows
   venv\Scripts\activate

   # macOS / Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
   *Alternatively, install core packages directly:*
   ```bash
   pip install streamlit librosa numpy scikit-learn joblib opencv-python folium streamlit-folium gTTS reportlab plotly requests pandas
   ```

4. **Launch the Control Tower**
   ```bash
   streamlit run app.py
   ```
   Access the interface locally at `http://localhost:8501`.

---

## Operational Workflow

1. **Fleet Configuration:** Select an active vehicle route, commodity type, cargo tonnage, and baseline market valuation in the control sidebar.
2. **Environmental Synchronization:** Query live route temperatures via GPS coordinates or simulate thermal spikes using the ambient temperature controls.
3. **Sensor Data Ingestion:**
   - Upload a `.wav` acoustic tap recording to evaluate internal cellular firmness.
   - Upload a `.jpg` or `.png` cargo image to inspect surface integrity and XAI defect overlays.
4. **Automated Escalation:** When calculated spoilage risk surpasses `45%`, review the updated GIS reroute map, secondary B2B liquidation bids, rescued revenue metrics, synthesized driver audio alerts, and exportable PDF compliance reports.

---

## Roadmap & Future Enhancements

- **Edge Model Quantization:** Convert acoustic and vision inference pipelines to TensorFlow Lite / ONNX formats for offline on-device mobile execution in low-connectivity corridors.
- **Signal Noise Suppression:** Implement digital Butterworth bandpass filtering (`100 Hz – 2000 Hz`) to isolate produce resonance frequencies from ambient engine and highway noise.
- **Automated Messaging Integration:** Integrate Twilio SMS/WhatsApp webhooks to transmit turn-by-turn reroute links directly to driver mobile devices.

---

## Author

**Sindhuja M**  
Department of Artificial Intelligence and Data Science  
Rajalakshmi Institute of Technology  
