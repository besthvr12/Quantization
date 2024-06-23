# PIDNetQAT
# Quantization Aware Training (QAT) Approaches

In Quantization Aware Training (QAT), we simulate the effects of quantization during the training process. This helps the model adapt to the reduced precision and improves its robustness when deployed on hardware with limited computational resources. We can apply QAT using the following three approaches:

## 1. Collect Stats and Calibration, Then Train the Model

In this approach, we perform two steps before starting the actual training:

1. **Collect Statistics**:
    - Run the model on a representative dataset to collect activation statistics.
    - This step helps in understanding the range and distribution of the activations.

2. **Calibration**:
    - Based on the collected statistics, calibrate the model to determine the optimal scaling factors for quantization.
    - Calibration ensures that the quantized model closely approximates the floating-point model.

After these steps, we train the model with the quantization effects simulated during the training process.

## 2. Collect Stats, No Calibration, and Train the Model

In this approach, we only collect statistics but skip the calibration step:

1. **Collect Statistics**:
    - Gather activation statistics from a representative dataset.

2. **No Calibration**:
    - Skip the calibration step and proceed directly to training.

By not performing calibration, we rely on the collected statistics to provide enough information for the quantization-aware training process.

## 3. No Calibration, Directly Train the Model

In this approach, we skip both statistics collection and calibration:

1. **No Statistics Collection**:
    - Do not gather activation statistics from the dataset.

2. **No Calibration**:
    - Skip calibration entirely.

Instead, we directly train the model with the quantization effects simulated during training. This method initializes the parameters and allows the model to adapt to quantization without prior knowledge of the data distribution.

