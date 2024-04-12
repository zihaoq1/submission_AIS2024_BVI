"""
Marcos Conde, 2024

AIS: Vision, Graphics and AI for Streaming CVPR 2024 Workshop
"""

import os
import torch
import time
import pathlib
import logging
import argparse
import numpy as np
import importlib
import sys
import datetime
import logging

import torch.nn.functional as F

import torch.nn as nn
import torchvision.models as models

from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import clogger
from model_summary import get_model_flops
from model_summary import get_model_complexity_info
#from ptflops import get_model_complexity_info

import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
#import torch
#import numpy as np
#import argparse

mean_stds = {
    "FasterVQA": (0.14759505, 0.03613452), 
    "FasterVQA-MS": (0.15218826, 0.03230298),
    "FasterVQA-MT": (0.14699507, 0.036453716),
    "FAST-VQA":  (-0.110198185, 0.04178565),
    "FAST-VQA-M": (0.023889644, 0.030781006), 
}

opts = {
    "FasterVQA": "./options/fast/f3dvqa-b.yml", 
    "FasterVQA-MS": "./options/fast/fastervqa-ms.yml", 
    "FasterVQA-MT": "./options/fast/fastervqa-mt.yml", 
    "FAST-VQA": "./options/fast/fast-b.yml", 
    "FAST-VQA-M": "./options/fast/fast-m.yml", 
}


class VQAModel(nn.Module):
    """
    Dummy VQA model.
    """
    def __init__(self, num_frames=60, height=1080, width=1920):
        super(VQAModel, self).__init__()
        # Initialize MobileNet
        self.mobilenet = models.mobilenet_v2(pretrained=True).features
        for param in self.mobilenet.parameters():
            param.requires_grad = False  # Freeze MobileNet parameters
        
        # Adaptive pooling to handle varying sizes
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        
        # Placeholder for the feature dimension. This needs to be adjusted based on the output of MobileNet.
        # For MobileNetV2, the feature dimension is 1280.
        feature_dim = 1280
        
        # Linear layer for quality prediction
        self.fc = nn.Linear(feature_dim, 1)  # Predicting a single score
        
    def forward(self, x):
        # x shape: [batch, frames, 3, H, W]
        batch_size, num_frames, C, H, W = x.shape
        
        # Process each frame individually
        x = x.view(batch_size * num_frames, C, H, W)  # Reshape for processing by MobileNet
        x = self.mobilenet(x) # [batch_size * num_frames, 1280, 60, 34]
        
        # Apply adaptive pooling
        x = self.pooling(x) # torch.Size([batch_size * num_frames, 1280, 1, 1])

        # Reshape back to (batch, frames, feature_dim)
        x = x.view(batch_size, num_frames, -1) # torch.Size([1, 30, 1280])
        
        # Average the features across the frames -- Simple feature aggregation
        x = torch.mean(x, dim=1) # torch.Size([1, 1280])
        
        # Predict the quality score
        x = self.fc(x)
        return x
    

def main(args, opt):

    """
    SETUP LOGGER
    """
    clogger.logger_info("AIS24-VQA", log_path=os.path.join(args.save_dir, f"Submission_{args.submission_id}.txt"))
    logger = logging.getLogger("AIS24-VQA")

    """
    BASIC SETTINGS
    """
    opt = opts.get(args.model, opts["FasterVQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    """
    LOAD MODEL
    """
    model = DiViDeAddEvaluator(**opt["model"]["args"])
    model = model.to(device)
    model.eval()
    
    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info(f"Results of {args.submission_id}")
    logger.info('Params number: {}'.format(number_parameters))
            
    """
    SETUP RUNTIME
    """
    test_results = OrderedDict()
    test_results["runtime"] = []
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    """
    TESTING
    """
    assert args.frames > 0
    ## fragment size
    input_dim = (1, 3, 32, 224, 224)
    input_data = torch.randn(input_dim).to(device)
    logger.info('Fragment size: {}'.format(input_data.shape))
    #input_dim = (1, args.frames, 3, args.imsize[0], args.imsize[1])
    #input_data = torch.randn(input_dim).to(device)
    #logger.info('Input resolution: {}'.format(input_data.shape))
    
    t_data_opt = opt["data"]["val-kv1k"]["args"]
    s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]

    if args.fp16:
        import torch_tensorrt
        input_data = input_data.half()
        model = model.half()
        if args.trt:
            model = torch_tensorrt.compile(model, inputs= [torch_tensorrt.Input(input_dim, dtype=torch_tensorrt.dtype.half)], enabled_precisions= {torch_tensorrt.dtype.half})

    # GPU warmp up
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(3):
            vsamples = {}
            for sample_type, sample_args in s_data_opt.items():
                vsamples[sample_type] = input_data
            _ = model(vsamples)
            
    print("Start timing ...")
    torch.cuda.synchronize()

    with torch.no_grad():
        for _ in tqdm(range(args.repeat)):       
            start.record()
            vsamples = {}
            for sample_type, sample_args in s_data_opt.items():
                vsamples[sample_type] = input_data
            _ = model(vsamples)
            end.record()

            torch.cuda.synchronize()
              
            test_results["runtime"].append(start.elapsed_time(end))  # milliseconds

        ave_runtime = sum(test_results["runtime"]) / len(test_results["runtime"])
        per_frame_time = (ave_runtime / args.batch_size)/args.frames

        logger.info(f"------> INPUT {args.imsize[0]}x{args.imsize[1]}, {args.frames} frames, {args.batch_size} clip")
        logger.info('------> Average runtime on clip {}-frames  of ({}) is : {:.6f} ms'.format(args.frames, args.submission_id, ave_runtime / args.batch_size ))
        logger.info('------> Average runtime per frame of ({}) is : {:.6f} ms'.format(args.submission_id, per_frame_time ))
        logger.info('------> Average FPS of ({}) is : {:.6f} FPS'.format(args.submission_id, 1000 / per_frame_time ))
        
        if not args.trt:
            ## fragment size
            #input_dim = (3, 32, 224, 224) #input_dim    = (args.frames, 3, args.imsize[0], args.imsize[1])
            #desired_macs = 2 * args.imsize[0] * args.imsize[1]
            #macs, params = get_model_complexity_info(model, input_dim, opt, as_strings=False,
            #                               print_per_layer_stat=False)
            
            # one MACs equals roughly two FLOPs
            # macs2 = get_model_flops(model, input_dim, print_per_layer_stat=False, verbose=False)
            # macs2 = macs2 / 10 ** 9

            #logger.info(f"------> MACs per clip {args.frames}-frames : {macs}")
            #logger.info(f"------> MACs per clip {args.frames}-frames : {macs / 10 ** 9 } [G]")
            #logger.info(f"------> MACs per frame : {macs / 10 ** 9 / args.frames } [G]")

            num_parameters = sum(map(lambda x: x.numel(), model.parameters()))
            num_parameters = num_parameters / 10 ** 6
            logger.info(f"------> #Params {num_parameters} [M]")

        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # specify submission
    parser.add_argument("--submission-id", type=str, default="test_model")
    
    # specify dirs
    parser.add_argument("--save-dir", type=str, default="submissions/")
    
    # specify test case
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1, help="Number of ?-frame clips")
    parser.add_argument("--frames", type=int, default=30, help="Number of frames. Default 30FPS.")
    parser.add_argument("--imsize", type=int, nargs="+", default=[1920, 1080], help="Frame resolution.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--trt", action="store_true")
    
    parser.add_argument(
        "-m", "--model", type=str, 
        default="FasterVQA", 
        help="model type: can choose between FasterVQA, FasterVQA-MS, FasterVQA-MT, FAST-VQA, FAST-VQA-M",
    )
    
    ## can be your own
    parser.add_argument(
        "-v", "--video_path", type=str, 
        default="./demos/10053703034.mp4", 
        help="the input video path"
    )
    
    parser.add_argument(
        "-d", "--device", type=str, 
        default="cuda", 
        help="the running device"
    )
    
    
    args = parser.parse_args()
    
    opt = opts.get(args.model, opts["FasterVQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)

    main(args, opt)