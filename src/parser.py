from collections import defaultdict
from typing import List, Tuple
from omegaconf import DictConfig, ListConfig
import torch.nn as nn


def parse_layer(layers: ListConfig, D: int, K: int, L: int) -> List[Tuple]:
    layers_tup = []
    for layer in layers:
        in_feature, out_feature, act_fn = layer.replace(' ', '').split(',')
        act_fn = getattr(nn, act_fn)
        in_feature = in_feature.replace('D', str(D)).replace('K', str(K)).replace('L', str(L))
        out_feature = out_feature.replace('D', str(D)).replace('K', str(K)).replace('L', str(L))
        layers_tup.append((int(in_feature), int(out_feature), act_fn()))
    return layers_tup


def parse_layers(cfg: DictConfig, D: int, K: int, L: int):
    layer_dict = defaultdict()
    for layer_name, layers in cfg.items():
        layer_dict[layer_name + '_layers'] = parse_layer(layers, D, K, L)
    return layer_dict


def parse_configs(cfg: DictConfig):
    return parse_layers(cfg.layers, cfg.D, cfg.K, cfg.L), cfg.D, cfg.L, cfg.K
