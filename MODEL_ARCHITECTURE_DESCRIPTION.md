# Mô tả Kiến trúc Model Contrastive Learning cho Time Series Anomaly Detection

## Tổng quan

Model này sử dụng kiến trúc **Contrastive Learning** kết hợp với **Transformer** và **Temporal Convolutional Network (TCN)** để phát hiện anomaly trong dữ liệu time series. Model được thiết kế để học biểu diễn robust của dữ liệu bình thường thông qua việc so sánh giữa dữ liệu gốc và dữ liệu được augment.

## Kiến trúc Tổng thể

```
Input Data (Original + Augmented)
           ↓
    ┌─────────────────┐
    │   Augmentation  │ ←─── Augmented Data
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │     Encoder     │ ←─── TCN + Transformer
    └─────────────────┘
           ↓
    ┌─────────────────┐
    │ augmented_encoded│ ←─── Encoder Output
    └─────────────────┘
           ↓
      ┌────┴────┐
      ↓         ↓
  ┌────────┐  ┌────────┐
  │Projection│ │Decoder │ ←─── 2 nhánh song song
  └────────┘  └────────┘
      ↓         ↓
  Contrastive  Reconstruction
      Loss       Loss
      ↓         ↓
    ┌─────────────────┐
    │   Total Loss    │
    └─────────────────┘
```

**Lưu ý quan trọng về luồng dữ liệu:**
- Từ `augmented_encoded` (output của Encoder), dữ liệu **tách ra thành 2 nhánh song song**:
  - **Nhánh 1**: `augmented_encoded` → `Projection MLP` → `augmented_projection` (dùng cho Contrastive Loss)
  - **Nhánh 2**: `augmented_encoded` → `Decoder MLP` → `reconstructed` (dùng cho Reconstruction Loss)
- **KHÔNG phải** là luồng tuần tự: `augmented_encoded` → `Projection` → `Decoder`

## Các Module Chi tiết

### 1. Augmentation Module (`src/modules/augmentation.py`)

**Chức năng**: Tạo ra các phiên bản augment của dữ liệu gốc để học contrastive learning.

**Kiến trúc**:
- **LinearAugmentation**: Linear layer với GELU activation
- **MLPAugmentation**: Multi-layer perceptron với feature flattening
- **CNNAugmentation**: 1D CNN với kernel size có thể điều chỉnh

**Cách hoạt động**:
```python
# Mỗi module tạo ra output cùng dimension với input
linear_out = self.linear_module(x)
cnn_out = self.cnn_module(x)
tcn_out = self.tcn_module(x)
lstm_out = self.lstm_module(x)
transformer_out = self.transformer_module(x)

# Kết hợp outputs với learned weights
probs = F.softmax(self.alpha / self.temperature, dim=0)
combined_output = weighted_sum_of_outputs
```

**Tham số quan trọng**:
- `temperature`: Điều khiển độ "sharp" của softmax weights (τ)
- `alpha`: Learned parameters cho việc kết hợp các augmentation
- `dropout`: Dropout rate cho regularization

### 2. Encoder Module (`src/modules/encoder.py`)

**Chức năng**: Encode dữ liệu thành biểu diễn high-level bằng cách kết hợp TCN và Transformer.

**Kiến trúc**:

#### TransformerEncoderBlock
```python
encoder_layer = TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    batch_first=True
)
self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=transformer_layers)
```

#### TCNBlock
```python
# Exponential dilation: 2^i cho mỗi layer
for i in range(num_layers):
    dilation = 2 ** i
    conv = nn.Conv1d(in_channels, output_dim, kernel_size, 
                     padding=(kernel_size-1)*dilation, dilation=dilation)
```

**Cách kết hợp**:
- **Concat method**: Concatenate TCN và Transformer outputs, sau đó project về d_model
- **Stack method**: Stack outputs và project về d_model

### 3. Decoder Module (`src/modules/decoder.py`)

**Chức năng**: Reconstruct dữ liệu gốc từ encoded representation của augmented data.

**Kiến trúc**:
```python
# Simple MLP Decoder
layers = []
for hidden_dim in hidden_dims:
    layers.extend([
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout)
    ])
layers.append(nn.Linear(input_dim, output_dim))
```

**Đặc điểm**:
- Không có memory mechanism (khác với LSTM/GRU)
- Architecture đơn giản để tránh overfitting
- Reconstruct từ augmented encoding về original data

### 4. Contrastive Model (`src/model/contrastive_model.py`)

**Chức năng**: Kết hợp tất cả các module và tính toán loss functions.

**Forward Pass**:
```python
def forward(self, original_data, augmented_data):
    # Apply augmentation
    augmented_data = self.augmentation(augmented_data)
    
    # Encode both
    original_encoded = self.encoder(original_data)      # (batch, seq_len, d_model)
    augmented_encoded = self.encoder(augmented_data)    # (batch, seq_len, d_model)
    
    # NHÁNH 1: Project for contrastive learning
    original_projection = self.projection_mlp(original_encoded)      # (batch, seq_len, projection_dim)
    augmented_projection = self.projection_mlp(augmented_encoded)    # (batch, seq_len, projection_dim)
    
    # NHÁNH 2: Reconstruct from augmented encoding (song song với nhánh 1)
    reconstructed = self.decoder(augmented_encoded)      # (batch, seq_len, input_dim)
    
    return {
        'original_encoded': original_encoded,
        'augmented_encoded': augmented_encoded,
        'original_projection': original_projection,       # Nhánh 1
        'augmented_projection': augmented_projection,     # Nhánh 1
        'reconstructed': reconstructed                    # Nhánh 2
    }
```

## Loss Functions

### 1. Contrastive Loss (InfoNCE-based)

**Công thức**:
```
L_Ci = -(1/n) * Σ_{i=1}^{n} log(exp(sim(α_i, α'_i)) / 
         (exp(sim(α_i, α'_i)) + Σ_{j≠i} (1 - sim(α_i, α_j) + ε) * exp(sim(α_i, α'_j))))
```

**Giải thích**:
- `α_i`: Projection của original data thứ i
- `α'_i`: Projection của augmented data thứ i  
- `sim(α_i, α'_i)`: Cosine similarity được chuyển về range [0,1]
- `ε`: Small constant để tránh numerical instability
- `j≠i`: Negative samples (các samples khác trong batch)

**Mục đích**: 
- Pull positive pairs (original, augmented của cùng sample) lại gần nhau
- Push negative pairs (original của sample này, augmented của sample khác) ra xa nhau
- Weighted by similarity để tránh hard negatives

### 2. Reconstruction Loss

**Công thức**:
```
L_R = MSE(original_data, reconstructed_data)
```

**Mục đích**: Đảm bảo model có thể reconstruct lại dữ liệu gốc từ augmented representation.

### 3. Total Loss

**Công thức**:
```
L_total = λ_c * L_contrastive + λ_r * L_reconstruction
```

**Tham số**:
- `λ_c` (contrastive_weight): Trọng số cho contrastive loss (default: 1.0)
- `λ_r` (reconstruction_weight): Trọng số cho reconstruction loss (default: 1.0)

## Chuẩn hóa Dữ liệu

### 1. Data Preprocessing (`src/utils/dataloader.py`)

**Các phương pháp chuẩn hóa**:

#### Z-score Normalization
```python
def z_score_normalize(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        normalized_data[i] = zscore(data[i], nan_policy='omit')
    return normalized_data
```

#### Min-Max Normalization
```python
def min_max_normalize(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        data_min = np.min(data[i])
        data_max = np.max(data[i])
        if data_max > data_min:
            normalized_data[i] = (data[i] - data_min) / (data_max - data_min)
    return normalized_data
```

#### Robust Normalization
```python
def robust_normalize(data):
    normalized_data = np.zeros_like(data)
    for i in range(data.shape[0]):
        median = np.median(data[i])
        q75, q25 = np.percentile(data[i], [75, 25])
        iqr = q75 - q25
        if iqr > 0:
            normalized_data[i] = (data[i] - median) / iqr
    return normalized_data
```

### 2. Dataset-specific Normalization

**ECG Dataset**:
- Format: `[feature1, feature2, label]` → Extract 2 features
- Global normalization to [0,1] using train data statistics

**PSM Dataset**:
- 25 features từ sensors
- Handle NaN values với forward/backward fill
- Global normalization to [0,1]

**UCR Dataset**:
- Single feature time series
- Handle NaN values với interpolation
- Global normalization to [0,1]

### 3. Contrastive Dataset Normalization

**Window-based Processing**:
```python
class ContrastiveDataset:
    def __getitem__(self, idx):
        # Extract window
        window_data = self.data[:, start_idx:end_idx]
        
        # Apply masking if specified
        if self.mask_mode == 'time':
            mask_idx = random.choice(time_indices)
            window_data[:, mask_idx] = 0.0
        elif self.mask_mode == 'feature':
            mask_idx = random.choice(feature_indices)
            window_data[mask_idx, :] = 0.0
        
        # Transpose to (window_size, features)
        return window_data.T
```

## Quy trình Training

### 1. Data Loading (`src/model/train_contrastive.py`)

```python
def create_contrastive_dataloaders():
    # Load dataset using existing dataloader
    dataloaders = create_dataloaders(...)
    
    # Create contrastive dataset
    contrastive_train_dataset = ContrastiveDataset(
        data=train_dataset.data,
        window_size=window_size,
        stride=1,  # Non-overlapping windows
        mask_mode=mask_mode,
        mask_ratio=mask_ratio
    )
    
    return train_dataloader, val_dataloader
```

### 2. Training Loop

```python
def train_epoch(self):
    for original_batch, augmented_batch in train_dataloader:
        # Forward pass
        losses = self.model.compute_total_loss(
            original_batch, augmented_batch,
            contrastive_weight=self.contrastive_weight,
            reconstruction_weight=self.reconstruction_weight,
            epsilon=self.epsilon
        )
        
        # Backward pass
        losses['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        
        # Update parameters
        self.optimizer.step()
```

### 3. Learning Rate Scheduling

**Các loại scheduler được hỗ trợ**:
- **Cosine**: `CosineAnnealingLR`
- **Step**: `StepLR` 
- **Exponential**: `ExponentialLR`
- **Plateau**: `ReduceLROnPlateau`

## Quy trình Testing/Inference

### 1. Model Loading (`src/inference/inference.py`)

```python
class ContrastiveInference:
    def load_model(self, model_path):
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model architecture
        model = ContrastiveModel(
            input_dim=self.input_dim,
            d_model=checkpoint['d_model'],
            projection_dim=checkpoint['projection_dim'],
            # ... other parameters
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
```

### 2. Sliding Window Inference

```python
def predict_anomalies(self, test_data, window_size, stride=1):
    predictions = []
    reconstruction_errors = []
    
    for i in range(0, len(test_data) - window_size + 1, stride):
        # Extract window
        window = test_data[i:i+window_size]
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(window, window)  # Same data for both inputs
            
            # Compute reconstruction error
            reconstruction_error = F.mse_loss(
                outputs['reconstructed'], window
            ).item()
            
            reconstruction_errors.append(reconstruction_error)
    
    # Convert reconstruction errors to anomaly scores
    anomaly_scores = np.array(reconstruction_errors)
    
    return anomaly_scores
```

### 3. Anomaly Detection Strategy

**Threshold-based Detection**:
```python
def detect_anomalies(self, anomaly_scores, threshold_method='percentile', threshold_value=95):
    if threshold_method == 'percentile':
        threshold = np.percentile(anomaly_scores, threshold_value)
    elif threshold_method == 'iqr':
        q75, q25 = np.percentile(anomaly_scores, [75, 25])
        threshold = q75 + 1.5 * (q75 - q25)
    
    predictions = (anomaly_scores > threshold).astype(int)
    return predictions, threshold
```

### 4. Evaluation Metrics

**Binary Classification Metrics**:
```python
def binary_classification_metrics(y_true, y_pred):
    tp = np.sum(y_true & y_pred)
    fp = np.sum(~y_true & y_pred)
    fn = np.sum(y_true & ~y_pred)
    tn = np.sum(~y_true & ~y_pred)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
    
    return accuracy, precision, recall, f1, (tp, fp, fn, tn)
```

**Point Adjustment**:
```python
def adjustment(gt, pred):
    # Expand predicted anomalies to cover entire ground truth anomaly region
    # when at least one point in the region is correctly predicted
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            # Expand backward and forward
            # ... expansion logic
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred
```

## Cấu hình Model

### Tham số chính:

**Architecture**:
- `input_dim`: Số features đầu vào (2 cho ECG, 1 cho PD, 25 cho PSM, etc.)
- `d_model`: Model dimension cho Transformer (default: 128)
- `projection_dim`: Dimension cho contrastive projection (default: 128)
- `nhead`: Số attention heads (default: 4)
- `transformer_layers`: Số transformer layers (default: 3)
- `tcn_kernel_size`: Kernel size cho TCN (default: 3)
- `tcn_num_layers`: Số TCN layers (default: 3)

**Training**:
- `window_size`: Kích thước sliding window (default: 128)
- `batch_size`: Batch size (default: 32)
- `learning_rate`: Learning rate (default: 1e-4)
- `contrastive_weight`: Trọng số contrastive loss (default: 1.0)
- `reconstruction_weight`: Trọng số reconstruction loss (default: 1.0)
- `temperature`: Temperature cho InfoNCE loss (default: 1)

**Augmentation**:
- `aug_nhead`: Nhead cho augmentation transformer (default: 2)
- `aug_num_layers`: Số layers cho augmentation transformer (default: 1)

## Điểm mạnh của Kiến trúc

1. **Multi-scale Feature Learning**: Kết hợp TCN (local patterns) và Transformer (global dependencies)
2. **Robust Augmentation**: 3 loại augmentation khác nhau với learned combination weights
3. **Contrastive Learning**: Học biểu diễn robust bằng cách so sánh positive/negative pairs
4. **Reconstruction Constraint**: Đảm bảo model không chỉ học contrastive mà còn có thể reconstruct
5. **Flexible Architecture**: Có thể điều chỉnh cho nhiều loại dataset khác nhau

## Hạn chế và Cải thiện

1. **Computational Cost**: Model khá phức tạp với nhiều components
2. **Hyperparameter Sensitivity**: Nhiều tham số cần tuning
3. **Memory Usage**: Contrastive learning cần nhiều memory cho large batches
4. **Interpretability**: Khó giải thích tại sao model đưa ra prediction

## Kết luận

Kiến trúc này là một approach mạnh mẽ cho time series anomaly detection, kết hợp các kỹ thuật state-of-the-art trong deep learning. Model học được biểu diễn robust của dữ liệu bình thường và có thể phát hiện anomalies thông qua reconstruction error và contrastive learning.
