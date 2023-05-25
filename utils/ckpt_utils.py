import torch
import os

def sanitize_ckpt(state_dict):
    new_dict = {}
    for k in state_dict.keys():
        if k.startswith("module."):
            new_k = k[7:]
            new_dict[new_k] = state_dict[k]
    return new_dict

def sanitize_CSI_ckpt(state_dict):
    new_dict = {}

    contrastive_head_dict = {}
    for k in state_dict.keys():
        if k.startswith('conv1') or k.startswith('bn1') or k.startswith('layer'):
            new_dict[k] = state_dict[k]
        if k.startswith('linear'):
            new_k = k.replace('linear', 'fc')
            new_dict[new_k] = state_dict[k]
        if k.startswith('simclr'):
            contrastive_head_dict[k] = state_dict[k]

    return {"backbone": new_dict, "head": contrastive_head_dict}

def sanitize_simclr_ckpt(state_dict):
    new_dict = {}

    bn_keys = ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']

    def sanitize_bn(old_dict, orig_base_key, new_dict, new_base_key):
        for k in bn_keys:
            new_dict[new_base_key+k] = old_dict[orig_base_key+k]  

        return new_dict

    def sanitize_conv_layer(old_dict, orig_base_key, new_dict, new_base_key, n_blocks=3):
        # first we sanitize the projection/downsample
        new_dict[new_base_key+'0.downsample.0.weight'] = old_dict[orig_base_key+'0.projection.shortcut.weight']
        new_dict = sanitize_bn(old_dict, orig_base_key+'0.projection.bn.0.', new_dict, new_base_key+'0.downsample.1.')

        for b in range(n_blocks):
            for i in range(3):
                new_dict[new_base_key+str(b)+'.conv'+str(i+1)+'.weight'] = old_dict[orig_base_key+str(b)+'.net.'+str(i*2)+'.weight']
                new_dict = sanitize_bn(old_dict, orig_base_key+str(b)+'.net.'+str(i*2+1)+'.0.', new_dict, new_base_key+str(b)+'.bn'+str(i+1)+'.')

        return new_dict 

    # preliminar layers:
    new_dict['conv1.weight'] = state_dict['net.0.0.weight']
    new_dict = sanitize_bn(state_dict, "net.0.1.0.", new_dict, "bn1.")

    # sanitize res layer 1
    new_dict = sanitize_conv_layer(state_dict, "net.1.blocks.", new_dict, "layer1.", n_blocks=3)

    # res layer 2 
    new_dict = sanitize_conv_layer(state_dict, "net.2.blocks.", new_dict, "layer2.", n_blocks=4)

    # res layer 3 
    new_dict = sanitize_conv_layer(state_dict, "net.3.blocks.", new_dict, "layer3.", n_blocks=23)

    # res layer 4
    new_dict = sanitize_conv_layer(state_dict, "net.4.blocks.", new_dict, "layer4.", n_blocks=3)

    # head 
    new_dict['fc.weight'] = state_dict['fc.weight']
    new_dict['fc.bias'] = state_dict['fc.bias']

    wrapper = {'backbone': new_dict}

    return wrapper

