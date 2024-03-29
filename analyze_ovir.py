from openvino.runtime import Core
from tqdm import tqdm
import torch
from collections import OrderedDict
from pathlib import Path
import numpy as np
from collections import Counter
import os

QDTYPE_SPECIAL_VALUES={
    'u4': [0, 1, 2, 4, 8],
    'u8': [0, 1, 2, 4, 8, 16, 32, 64, 128],
    'int8': [-1, -2, -4, -8, -16, -32, -64, -128, 0, 1, 2, 4, 8, 16, 32, 64]
}

zero_point_map = {
    'u4': 8,
    'u8': 128,
    'int8': 0,
}

def get_uniq_value_stats(tensor, q_dtype):
    if q_dtype not in QDTYPE_SPECIAL_VALUES.keys():
        raise NotImplementedError(f"Unsupported q_dtype {q_dtype}")
    
    value_counts = Counter(tensor.flatten()) 
    total_elements = sum(value_counts.values())
    
    top1_val, top1_count = value_counts.most_common(1)[0]
    top1_tuple = (top1_val, top1_count/total_elements) 

    # Calculate ratio for each value
    count_ratio_dict = {value: {'count': count, 'ratio': count / total_elements}
                        for value, count in value_counts.items()}
    
    # # Find unique elements and their counts
    # unique_values, counts = np.unique(tensor, return_counts=True)
    # # Calculate the total number of elements in the tensor
    # total_elements = tensor.size
    # # Calculate the relative ratio for each unique value
    # ratios = counts / total_elements

    special_value_count = 0
    special_value_ratio = 0
    sparsity = 0
    zero_count = 0
    
    # for value, count, ratio in zip(unique_values, counts, ratios):
    for value, vdict in count_ratio_dict.items():
        count = vdict['count']
        ratio = vdict['ratio']
        if value == zero_point_map[q_dtype]:
            sparsity = ratio
            zero_count = count
        
        # zero will enter both above and below
        if value in QDTYPE_SPECIAL_VALUES[q_dtype]:
            special_value_count += count
            special_value_ratio += ratio
    
    return dict(
        numel=total_elements,
        sparsity=sparsity,
        special_value_ratio=special_value_ratio,
        top1=top1_tuple,
        raw=count_ratio_dict
    )


def get_ir_pair(model_dir):
    p = Path(model_dir)
    return p/"openvino_model.xml", p/"openvino_model.bin"


# fc_numel = {
#     'llama-2-chat-7b ': {'min': 16777216, 'max': 45088768},
#     'mistral-7b ': {'min': 4194304, 'max': 58720256},
#     'gemma-2b-it': {'min': 524288, 'max': 33554432},
# }

fc_numel = {
    'llama-2-chat-7b': [16777216, 45088768],
    'mistral-7b': [4194304, 16777216, 58720256],
    'gemma-2b-it': [524288, 4194304, 33554432],
}

ovir_folder = "stable-diffusion-pokemons-1-5-quantized/unet"

# model_key = compressed_weight_folder.split("/")[2]

ir_xml, ir_bin = get_ir_pair(ovir_folder)

ie = Core()
ir_model = ie.read_model(ir_xml)

model_params = OrderedDict()

csv_path = os.path.join(ovir_folder, "weight_dist.csv")

with open(csv_path, "w") as outfile:
    outfile.write("layer,dtype,numel,sparsity,special_val_ratio,top1_val_ratio,top1_val\n")

    # for op in tqdm(ir_model.get_ordered_ops()):
    for op in ir_model.get_ordered_ops():
        if 'constant' in str(op.get_type_info()).lower():
            shape = tuple(op.get_output_shape(0))
            numel = np.prod(shape)
            

            if op.data.dtype.name == "int8":
                # print(f"{numel:15} | {str(shape):20} | {op.get_name():20} | {op.data.dtype.name}")
                layer = op.get_name()
                q_dtype = op.data.dtype.name
                
                # model_params[layer] = {}

                statdict = get_uniq_value_stats(op.data, op.data.dtype.name)

                # print("joto")
                # q_mode = "sym" if attr['q_zero_point'][0] == zero_point_map[attr['q_dtype']] else "asym"
                # is_top1_zero_point = "zero_point" if statdict['top1'][0] == zero_point_map[attr['q_dtype']] else statdict['top1'][0] # zero point is per channel per group
                # print(f"{layer:30} | {attr['q_dtype']} ({q_mode:>5}) | orig. shape: {str(attr['original_shape']):15} | numel: {statdict['numel']:>15,} | sparsity: {statdict['sparsity']:.2f} | special ratio: {statdict['special_value_ratio']:.2f} | top1 ratio: {statdict['top1'][1]:.2f} ({is_top1_zero_point:>10}) |")
                print(f"{layer:30} | {q_dtype} | orig. shape: {str(shape):20} | numel: {statdict['numel']:>15,} | sparsity: {statdict['sparsity']:.2f} | special ratio: {statdict['special_value_ratio']:.2f} | top1 ratio: {statdict['top1'][1]:.2f} (val: {statdict['top1'][0]})")
                
                outfile.write(f"{layer:>25},{q_dtype},{statdict['numel']:>15},{statdict['sparsity']:.4f},{statdict['special_value_ratio']:.4f},{statdict['top1'][1]:.4f},{statdict['top1'][0]}\n")

print('Done!')