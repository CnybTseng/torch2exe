import argparse
import math

def optsize(image_size: tuple = (1920, 1080),
    net_size_base: int = 1280, downsample: int = 32, fix_dim: int = 0
):
    net_size = [net_size_base, net_size_base]
    grid_size = [size / downsample for size in net_size]
    
    adjust_dim = 0 if fix_dim == 1 else 1
    net_size[adjust_dim] = (net_size[fix_dim] / image_size[fix_dim])* image_size[adjust_dim]
    grid_size[adjust_dim] = net_size[adjust_dim] / downsample
    
    grid_size_down = [math.floor(size) for size in grid_size]
    net_size_down = [size * downsample for size in grid_size_down]
    scales_down = [net_size_down[d] / image_size[d] for d in [0, 1]]
    scale_down = min(scales_down)
    net_size_scaled_down = [int(scale_down * size) for size in image_size]
    wasted_ratios_down = [(net_size_down[d] - net_size_scaled_down[d]) / net_size_down[d] for d in [0, 1]]
    wasted_ratio_down = max(wasted_ratios_down)
    save_ratios_down = [(net_size_base - size) / net_size_base for size in net_size_down]
    save_ratio_down = max(save_ratios_down)
    print(f'net_size_down: {net_size_down}')
    print(f'wasted_ratio_down: {wasted_ratio_down}, save_ratio_down: {save_ratio_down}')
    
    grid_size_up = [math.ceil(size) for size in grid_size]
    net_size_up = [size * downsample for size in grid_size_up]
    scales_up = [net_size_up[d] / image_size[d] for d in [0, 1]]
    scale_up = min(scales_up)
    net_size_scaled_up = [int(scale_up * size) for size in image_size]
    wasted_ratios_up = [(net_size_up[d] - net_size_scaled_up[d]) / net_size_up[d] for d in [0, 1]]
    wasted_ratio_up = max(wasted_ratios_up)
    save_ratios_up = [(net_size_base - size) / net_size_base for size in net_size_up]
    save_ratio_up = max(save_ratios_up)
    print(f'net_size_up: {net_size_up}')
    print(f'wasted_ratio_up: {wasted_ratio_up}, save_ratio_up: {save_ratio_up}')

if __name__ == '__main__':    
    optsize()