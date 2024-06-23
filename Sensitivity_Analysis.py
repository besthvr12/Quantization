import torch
import logging
import torch.nn as nn
import torch.optim as optim
import collections
import json
from pytorch_quantization import calib
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.tensor_quant import QuantDescriptor
from pytorch_quantization import quant_modules
from my_model import get_quantized_model, get_data_loader, get_optimizer, get_criterion
def eval_model(train_loader,model,optimizer,criterion):
    model.eval()
    total_loss = 0
    for inputs,targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs,targets)
        loss.backward()
        optimizer.step()
        total_loss +=loss.item()
    return total_loss/len(train_loader)


def sensitivity_analysis(model,train_loader,optimizer,criterion,tolerance=0.2):
    quant_layer_names=[]
    for name,module in model.named_modules():
        if isinstance(module,quant_nn.TensorQuantizer):
            quant_layer_names.append(name)
            print(name)

    
    logging.info(f"{len(quant_layer_names)} quantized layers found.")
    quant_layer_sensitivity={}
    for quant_layer in quant_layer_names:
        logging.info(f"Disable {quant_layer} to check sensitivity.")
        print(f"Disable {quant_layer} to check sensitivity.")

        for name,module in model.named_modules():
            if name==quant_layer:
                module.disable()
        performance = eval_model(train_loader,model,optimizer,criterion)
        logging.info(f"Performance: {performance}")
        print(f"Performance: {performance}")
        quant_layer_sensitivity[quant_layer] = tolerance - performance
        for name, module in model.named_modules():
            if name==quant_layer:
                module.enable()
    quant_layer_sensitivity = collections.OrderedDict(sorted(quant_layer_sensitivity.items(),key=lambda x:x[1],reverse=True))
    skipped_layers=[]
    for quant_layer, sensitivity in quant_layer_sensitivity.items():
        for name, module in model.named_modules():
             if name == quant_layer:
                logging.info(f"Disable {name}")
                print(f"Disable {name}")
                module.disable()
                skipped_layers.append(quant_layer)
        performance = eval_model(train_loader, model, optimizer, criterion)
        if performance <= tolerance:
            logging.info(f"Tolerance {tolerance} is met by skipping {len(skipped_layers)} sensitive layers.")
            print(f"Number of layers skipped to meet tolerance: {len(skipped_layers)} sensitive layers.")
            torch.save(model.state_dict(), "Best_Model_After_Sensitivity_Analysis.pt")
            break
    return quant_layer_sensitivity, skipped_layers


if __name__ == "__main__":
    import argparse
    

    parser = argparse.ArgumentParser(description='Sensitivity Analysis for Quantized Model')
    parser.add_argument('--model', type=str, required=True, help='Path to the quantized model')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--output', type=str, default='Best_Model_After_Sensitivity_Analysis.pt', help='Output file for the best model after sensitivity analysis')
    parser.add_argument('--tolerance', type=float, default=0.09, help='Performance tolerance for sensitivity analysis')
    args = parser.parse_args()

    model = get_quantized_model(args.model)
    train_loader = get_data_loader(args.data)
    optimizer = get_optimizer(model)
    criterion = get_criterion()

    quant_layer_sensitivity, skipped_layers = sensitivity_analysis(model, train_loader, optimizer, criterion, args.tolerance)

    with open(args.output.replace('.pt', '_sensitivity_results.json'), 'w') as f:
        json.dump(quant_layer_sensitivity, f, indent=4)

    print(f'Sensitivity analysis completed. Results saved to {args.output.replace(".pt", "_sensitivity_results.json")}')
    torch.save(model.state_dict(), args.output)
    print(f'Model saved to {args.output}')
