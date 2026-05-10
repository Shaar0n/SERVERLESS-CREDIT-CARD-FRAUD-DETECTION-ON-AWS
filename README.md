# Serverless Credit Card Fraud Detection on AWS

A machine learning solution for detecting fraudulent credit card transactions using a serverless architecture on AWS. The project trains an XGBoost classifier on historical transaction data and deploys it as a scalable, cost-effective inference endpoint using SageMaker and Lambda.

## 📋 Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Model](#model)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Performance Metrics](#performance-metrics)
- [Deployment](#deployment)
- [API Integration](#api-integration)
- [Monitoring](#monitoring)
- [Technologies](#technologies)
- [Files Description](#files-description)

## 🎯 Overview

This project implements a serverless fraud detection system that:
- Trains an XGBoost machine learning model on credit card transaction data
- Deploys the model on AWS SageMaker for real-time inference
- Integrates with API Gateway and Lambda for serverless predictions
- Monitors model performance through CloudWatch metrics
- Optimizes classification threshold for maximum F1 score

**Key Features:**
- ✅ Real-time fraud detection at scale
- ✅ Serverless architecture (no servers to manage)
- ✅ Cost-optimized with pay-per-use pricing
- ✅ Automated threshold tuning
- ✅ Comprehensive performance monitoring
- ✅ Production-ready inference code

## 🏗️ Architecture

```
┌─────────────┐
│   S3        │ (Data storage & model artifacts)
│  Bucket     │
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────┐
│  SageMaker Training Job         │ (XGBoost model training)
│  (Jupyter Notebook)             │
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  SageMaker Inference Endpoint   │ (Model deployment)
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  API Gateway                    │ (HTTP interface)
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  Lambda Function                │ (Serverless compute)
└──────┬──────────────────────────┘
       │
       ▼
┌─────────────────────────────────┐
│  CloudWatch                     │ (Monitoring & logging)
└─────────────────────────────────┘
```

## 📊 Dataset

- **Name:** Credit Card Fraud Detection Dataset
- **File:** `creditcard.csv`
- **Format:** CSV with 31 features (28 anonymized PCA components + Time, Amount, Class)
- **Target Variable:** `Class` (0 = Legitimate, 1 = Fraudulent)
- **Imbalanced:** Heavy class imbalance (mostly legitimate transactions)
- **Source:** S3 bucket (`fraud-ml-25249347-eu-north-1/raw/creditcard.csv`)

## 🤖 Model

### Model Type
**XGBoost Classifier** - Gradient boosting algorithm optimized for binary classification

### Hyperparameters
```python
XGBClassifier(
    n_estimators=250,           # Number of boosting rounds
    max_depth=6,                # Maximum tree depth
    learning_rate=0.08,         # Step size shrinkage
    subsample=0.9,              # Subsample ratio of training instances
    colsample_bytree=0.9,       # Subsample ratio of features
    objective="binary:logistic", # Binary classification
    eval_metric="aucpr",        # Precision-recall AUC for evaluation
    scale_pos_weight=scale_pos_weight,  # Weight adjustment for class imbalance
    random_state=42,            # Reproducibility
    n_jobs=-1                   # Use all available cores
)
```

### Training Details
- **Train/Test Split:** 80/20 stratified split
- **Objective Function:** Binary Logistic Loss
- **Class Weighting:** Automatically balanced using `scale_pos_weight` to handle class imbalance

## 📁 Project Structure

```
.
├── README.md                              # This file
├── creditcard.csv                         # Input dataset
├── fraud_implementation.ipynb             # Main Jupyter notebook with full pipeline
│
├── artifacts/                             # Generated model artifacts
│   ├── fraud_xgb_model.joblib            # Trained XGBoost model
│   ├── metrics.json                       # Model performance metrics
│   ├── feature_cols.json                  # Feature names/order
│   ├── confusion_matrix.csv              # Confusion matrix from test set
│   └── classification_report.csv         # Detailed classification metrics
│
├── inference.py                           # Lambda/SageMaker inference script
│
└── screenshots/                           # AWS console screenshots
    ├── API_GATEWAY_ROUTES.png            # API Gateway configuration
    ├── API_GATEWAY_STAGE.png             # API deployment stage
    ├── API_LAMBDA_METRICS.png            # Lambda invocation metrics
    ├── CloudWatch_API_GW_Metrics.png    # CloudWatch monitoring dashboard
    ├── Lambda_code_tab.png               # Lambda function code
    ├── Lamda_Test_result.png            # Lambda test invocation results
    ├── Live_Api_Response.png             # Live API response example
    ├── S3_Artifacts_folder.png          # S3 storage structure
    ├── Saved_Artifacts_Op.png           # Model artifacts in S3
    ├── Test_Upd.png                      # Test update operations
    ├── Threshold_Tuning.png              # Threshold optimization results
    └── Training.png                      # Model training progress
```

## 🚀 Getting Started

### Prerequisites
- AWS Account with appropriate IAM permissions
- SageMaker notebook instance or local Jupyter environment
- Python 3.7+
- Required Python packages (see cell 1 of notebook)

### Installation

1. **Clone or download the project:**
   ```bash
   cd SERVERLESS\ CREDIT\ CARD\ FRAUD\ DETECTION\ ON\ AWS
   ```

2. **Upload dataset to S3:**
   ```bash
   aws s3 cp creditcard.csv s3://fraud-ml-25249347-eu-north-1/raw/
   ```

3. **Install dependencies:**
   ```bash
   # From within the notebook or terminal
   pip install xgboost boto3 sagemaker scikit-learn pandas numpy joblib
   ```

4. **Configure AWS credentials:**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, region, and output format
   ```

## 💻 Usage

### Training & Model Development

1. **Open the Jupyter Notebook:**
   ```bash
   jupyter notebook fraud_implementation.ipynb
   ```

2. **Run cells sequentially:**
   - **Cell 1:** Install dependencies
   - **Cell 2:** Configure AWS settings (REGION, BUCKET, KEY)
   - **Cell 3:** Load dataset from S3
   - **Cell 4:** Prepare training/test split
   - **Cell 5:** Train XGBoost model
   - **Cell 6:** Tune decision threshold
   - **Cell 7:** Save model artifacts
   - **Cell 8-9:** Create inference script
   - **Cell 10:** Direct local inference testing
   - **Cell 11:** Deploy to SageMaker endpoint
   - **Cell 12:** Clean up resources (endpoint deletion)

### Making Predictions

#### Local Inference (Direct Model Usage)
```python
import joblib
import json

# Load model and metadata
model = joblib.load("artifacts/fraud_xgb_model.joblib")
with open("artifacts/metrics.json", "r") as f:
    metrics = json.load(f)

threshold = metrics["threshold"]

# Make prediction
probs = model.predict_proba(sample_data)[:, 1]
fraud_score = float(probs[0])
is_fraud = int(fraud_score >= threshold)

print(f"Fraud Probability: {fraud_score:.4f}")
print(f"Predicted Label: {is_fraud} (threshold: {threshold})")
```

#### SageMaker Endpoint Inference
```python
from sagemaker.sklearn.model import SKLearnModel
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer

# Load predictor
predictor = model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="fraud-xgb-endpoint"
)
predictor.serializer = CSVSerializer()
predictor.deserializer = JSONDeserializer()

# Make prediction
result = predictor.predict(sample_data_csv)
print(result)
# Output: {"fraud_probability": 0.95, "predicted_label": 1, "threshold_used": 0.52}
```

#### API Gateway & Lambda
```bash
curl -X POST https://your-api-endpoint/prod/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [...]}'
```

## 📈 Performance Metrics

The model is evaluated using multiple metrics optimized for imbalanced fraud detection:

| Metric | Value | Notes |
|--------|-------|-------|
| **Precision** | ~0.85 | Of predicted fraud, ~85% are actually fraudulent |
| **Recall** | ~0.80 | Catches ~80% of actual fraudulent transactions |
| **F1 Score** | ~0.82 | Harmonic mean balances precision and recall |
| **ROC-AUC** | ~0.97 | Excellent discrimination ability |
| **PR-AUC** | ~0.88 | Strong precision-recall trade-off |
| **Optimal Threshold** | ~0.52 | Tuned for maximum F1 score |

**Artifacts Generated:**
- `confusion_matrix.csv` - True/False Positives and Negatives
- `classification_report.csv` - Per-class precision, recall, F1
- `metrics.json` - Summary metrics and optimal threshold

## 🌐 Deployment

### SageMaker Endpoint Deployment

```python
from sagemaker.sklearn.model import SKLearnModel

sk_model = SKLearnModel(
    model_data="s3://fraud-ml-25249347-eu-north-1/models/fraud-xgb/model.tar.gz",
    role=role,
    entry_point="inference.py",
    source_dir="/tmp/sm_code",
    framework_version="1.2-1",
    py_version="py3"
)

predictor = sk_model.deploy(
    initial_instance_count=1,
    instance_type="ml.m5.large",
    endpoint_name="fraud-xgb-endpoint-20260422185436",
    wait=True
)
```

**Instance Type Options:**
- `ml.m5.large` - Recommended for testing (current)
- `ml.m5.xlarge` - For higher throughput
- `ml.m5.2xlarge` - For production scale

### API Gateway Integration

1. Create REST API in API Gateway
2. Create POST method → Lambda function
3. Lambda invokes SageMaker endpoint
4. Configure CORS and throttling
5. Deploy to "prod" stage

**Example Lambda Handler:**
```python
import json
import boto3
import csv
import io

sm_runtime = boto3.client('sagemaker-runtime')

def lambda_handler(event, context):
    body = json.loads(event['body'])
    features = body['features']
    
    csv_string = ','.join(map(str, features))
    
    response = sm_runtime.invoke_endpoint(
        EndpointName='fraud-xgb-endpoint-20260422185436',
        ContentType='text/csv',
        Body=csv_string
    )
    
    result = json.loads(response['Body'].read())
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }
```

### Cleanup

```python
# Delete endpoint after deployment
sm.delete_endpoint(EndpointName="fraud-xgb-endpoint-20260422185436")
sm.delete_endpoint_config(EndpointConfigName="fraud-xgb-endpoint-20260422185436")
```

## 📊 Monitoring

### CloudWatch Metrics

Monitored metrics include:
- **API Gateway Metrics:**
  - Request count
  - Latency (p50, p90, p99)
  - Error rates (4xx, 5xx)
  - Throttled requests

- **Lambda Metrics:**
  - Invocations
  - Duration
  - Errors
  - Concurrent executions
  - DLQ sends

- **SageMaker Metrics:**
  - Model latency
  - Invocation count
  - 4xx/5xx errors

### Creating Custom Dashboards

```python
# Create CloudWatch dashboard via notebook
import boto3

cloudwatch = boto3.client('cloudwatch')
# Define custom metrics and visualizations
```

### Logs

- **API Gateway Logs:** Access logs with request/response details
- **Lambda Logs:** CloudWatch Logs with execution details
- **SageMaker Logs:** Model container logs and invocations

## 🛠️ Technologies

| Component | Technology |
|-----------|-----------|
| **ML Framework** | XGBoost |
| **Training** | SageMaker Notebook |
| **Model Serialization** | joblib |
| **Deployment** | SageMaker Inference Endpoints |
| **Compute** | AWS Lambda |
| **API** | API Gateway |
| **Storage** | S3 |
| **Monitoring** | CloudWatch |
| **Language** | Python 3.7+ |
| **Data Science** | pandas, scikit-learn, numpy |

## 📄 Files Description

### Main Implementation
- **fraud_implementation.ipynb** - Complete end-to-end ML pipeline with all steps

### Model Artifacts
- **fraud_xgb_model.joblib** - Trained XGBoost model (binary serialized)
- **metrics.json** - Model performance metrics and optimal threshold
- **feature_cols.json** - List of input features in correct order
- **confusion_matrix.csv** - Test set confusion matrix
- **classification_report.csv** - Detailed classification metrics

### Inference Code
- **inference.py** - SageMaker-compatible inference script with:
  - `model_fn()` - Model loading
  - `input_fn()` - CSV input parsing
  - `predict_fn()` - Inference with threshold
  - `output_fn()` - JSON output formatting

### Dataset
- **creditcard.csv** - Raw credit card transaction data

### Documentation
- **README.md** - This comprehensive guide

### Screenshots
- API Gateway, Lambda, CloudWatch configurations
- Test results and live API responses
- Model training and threshold tuning visualizations

## 🔒 Security Considerations

1. **IAM Roles:** Use least-privilege IAM policies
2. **S3 Buckets:** Enable versioning and encryption
3. **API Authentication:** Implement API keys or AWS Signature
4. **Endpoint Access:** Use VPC endpoints or security groups
5. **Secrets Management:** Store sensitive config in AWS Secrets Manager
6. **Model Encryption:** Enable encryption at rest for S3 and SageMaker

## 💰 Cost Optimization

- **SageMaker Instances:** Use spot instances for training
- **Inference:** Scale down when not in use or use auto-scaling
- **Lambda:** First 1M requests free per month
- **Data Transfer:** Minimize S3 to SageMaker data transfer
- **Monitoring:** CloudWatch has free tier limits

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not found in S3 | Verify bucket name and run `s3.upload_file()` first |
| SageMaker endpoint timeout | Check instance type and VPC configuration |
| Lambda invocation fails | Verify IAM role has SageMaker:InvokeEndpoint permission |
| Inference predictions inconsistent | Ensure feature order matches training (use feature_cols.json) |
| High latency | Consider larger instance type (m5.xlarge, m5.2xlarge) |

## 📚 Further Improvements

- [ ] Implement A/B testing for model versions
- [ ] Add feature importance analysis
- [ ] Implement drift detection for model monitoring
- [ ] Create automated retraining pipeline
- [ ] Add explainability (SHAP values)
- [ ] Implement multi-armed bandit for threshold optimization
- [ ] Add caching layer for frequent predictions
- [ ] Implement batch inference for high-volume processing

## 📞 Support & Contribution

For issues or improvements:
1. Review CloudWatch logs for error details
2. Check SageMaker endpoint status
3. Verify S3 bucket permissions
4. Test locally with `fraud_xgb_model.joblib` before deploying

## 📜 License

This project is provided as-is for educational and commercial purposes.

## 📖 References

- [AWS SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)

---

**Created:** 2026-04-22  
**Last Updated:** 2026-05-10  
**Status:** Production Ready ✅
