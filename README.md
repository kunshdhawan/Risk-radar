# Risk Radar: Malware Analysis Tool

An intelligent, machine-learning powered Malware Analysis Tool designed to detect, classify, and analyze potentially malicious files through automated feature extraction and behavioral analysis.

# Overview
- In today’s evolving cybersecurity landscape, malware remains one of the most persistent threats. Traditional signature-based detection systems struggle against modern obfuscated, and polymorphic malware.

# Risk Radar addresses this challenge by combining:

 - Feature-based malware detection
 - Machine learning classification (XGBoost with SMOTE balancing)
 - Probability-based threat assessment
 - Interactive Flask web interface
 - PE and ZIP file inspection

# Supports:
 - PE files (.exe, .dll)
 - ZIP archives (including embedded malware)

# Detects the following categories:
 - Ransomware
 - Trojan
 - Worm
 - Virus
 - Spyware
 - Adware
 - Benign files

# Extracted features include:
 - File entropy
 - File size
 - Section count
 - Import count
 - DLL count
 - API calls
 - Network operations
 - Cryptographic indicators
 - Packed detection
 - Archive properties (for ZIP files)

# System Architecture
 User Upload
     ↓
 Flask Web Interface
     ↓
 MalwareTester Class
     ↓
 Feature Extraction
     ↓
 Feature Scaling
     ↓
 Machine Learning Prediction
     ↓
 Result Display and Visualization


# Project Structure
Risk-Radar/
│
├── app.py                     # Flask Web Application
├── malware_analysis.py        # MalwareTester class
├── train_model.py             # Model training pipeline
│
├── templates/
│   ├── main.html
│   └── how_it_works.html
│
├── static/
│   └── logo_bg.png
│
├── uploads/
│
├── enhanced_malware_model.pkl
├── feature_importance.png
├── confusion_matrix.png
└── training_report.txt

# Requirements
 - Software Requirements:-
  - Python 3.8 or higher
  - Flask
  - pefile
  - liefj
  - oblib
  - numpy
  - matplotlib
  - xgboost
  - scikit-learn
  - imbalanced-learn
  - tabulate

 - Hardware Requirements
  - Minimum 16 GB RAM
  - Intel i7 processor or higher recommended
  - Virtualization support (if extended sandboxing is implemented)

# How to Run
1. Train the Model (Optional)
  - python train_model.py
  - This generates the model file, evaluation metrics, and training report.

2. Run the Web Application
  - python app.py
  - Open your browser and navigate to: http://127.0.0.1:5000/
  - Upload a file to analyze it.

3. Command Line Usage
  - python malware_analysis.py enhanced_malware_model.pkl sample.exe

# Outputs:
 - For each analyzed file, the tool provides:
  - SHA256 hash
  - File entropy
  - File type
  - Predicted malware class
  - Confidence score
  - Malicious or Clean verdict
  - Feature breakdown
  - Probability distribution chart
  - JSON report

# Future Improvements
 - Full dynamic sandbox execution environment
 - Live network traffic monitoring
 - IOC extraction
 - REST API integration
 - Docker-based deployment
 - Cloud-based analysis
