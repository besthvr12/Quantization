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

# Sensitivity Analysis
In addition to the above approaches, we also performed sensitivity analysis. Using sensitivity analysis, we can identify which layers are most affected by loss. This information helps us selectively quantize layers that are less sensitive to quantization, thereby preserving the overall model performance.

I did provide a script for sensitivity analysis. Using this script we can do sensitivity analysis for any quantized model.
### How to Use

1. **Clone the repository:**

    ```bash
    https://github.com/besthvr12/PIDNetQAT.git
    cd PIDNetQAT
    ```

2. **Prepare your model and dataset:**

    Implement `get_model`, `get_data_loader`, `get_optimizer`, and `get_criterion` functions in a Python file (e.g., `my_model.py`). These functions should return your quantized model, a data loader for your dataset, an optimizer, and the loss criterion, respectively.
   
4. **Create a Configuration File:**

    Create a `config.yaml` file with the following content:

    ```yaml
    model_path: "path/to/quantized_model.pth"
    data_path: "path/to/dataset"
    output_file: "Best_Model_After_Sensitivity_Analysis.pt"
    tolerance: 0.09
    ```

5. **Run Sensitivity Analysis:**

    Use the `sensitivity_analysis.py` script with the configuration file to perform sensitivity analysis on your model.

    ```bash
    python sensitivity_analysis.py --config config.yaml
    ```

6. **View Results:**

    The sensitivity analysis results will be saved in a JSON file, and the best model will be saved as specified in the configuration file.

---

By following these steps, we can easily perform sensitivity analysis on the quantized models using a configuration file.



