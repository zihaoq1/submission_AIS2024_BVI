
import yaml
import decord
from fastvqa.datasets import get_spatial_fragments, SampleFrames, FragmentSampleFrames
from fastvqa.models import DiViDeAddEvaluator
import torch
import numpy as np
import argparse

import os
import pandas as pd

ignore_split = '_crf'
multiplier = 5

def sigmoid_rescale(score, model="FasterVQA"):
    mean, std = mean_stds[model]
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score

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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ### can choose between
    ### options/fast/f3dvqa-b.yml
    ### options/fast/fast-b.yml
    ### options/fast/fast-m.yml
    
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
    
    parser.add_argument(
        "-c", "--csv_path", type=str, 
        default="none", 
        help="the path to the csv file containing the list of test videos"
    )
    
    parser.add_argument(
        "-o", "--output_path", type=str, 
        default="./output.csv", 
        help="output path"
    )
    
    
    args = parser.parse_args()

    d_score = dict()
    if os.path.isfile(args.csv_path):
        df = pd.read_csv(args.csv_path)
        col_video = list(df['video'])
        for video_name in col_video:
            d_score[video_name] = -1
        #col_score = df['Prediction']
    else:
        col_video = []
    
    opt = opts.get(args.model, opts["FAST-VQA"])
    with open(opt, "r") as f:
        opt = yaml.safe_load(f)

    ### Model Definition
    evaluator = DiViDeAddEvaluator(**opt["model"]["args"]).to(args.device)
    evaluator.load_state_dict(torch.load(opt["test_load_path"], map_location=args.device)["state_dict"])

    ### Data Definition
    video_list = []
    if os.path.isdir(args.video_path):
        video_list = os.listdir(args.video_path)
    elif args.video_path.split('.')[-1] in ['mp4', 'avi']:
        video_list.append(args.video_path)
    else:
        print('Error: input video_path is not a video nor video directory!!')
    for video_name in video_list:
        if len(video_list) <= 1:
            video_path = args.video_path
        else:
            video_path = os.path.join(args.video_path, video_name)
        video_reader = decord.VideoReader(video_path)
        vsamples = {}
        t_data_opt = opt["data"]["val-kv1k"]["args"]
        s_data_opt = opt["data"]["val-kv1k"]["args"]["sample_types"]
        for sample_type, sample_args in s_data_opt.items():
            ## Sample Temporally
            if t_data_opt.get("t_frag",1) > 1:
                sampler = FragmentSampleFrames(fsize_t=sample_args["clip_len"] // sample_args.get("t_frag",1),
                                               fragments_t=sample_args.get("t_frag",1),
                                               num_clips=sample_args.get("num_clips",1),
                                              )
            else:
                sampler = SampleFrames(clip_len = sample_args["clip_len"], num_clips = sample_args["num_clips"])
            
            num_clips = sample_args.get("num_clips",1)
            frames = sampler(len(video_reader))
            print("Sampled frames are", frames)
            frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
            imgs = [frame_dict[idx] for idx in frames]
            video = torch.stack(imgs, 0)
            video = video.permute(3, 0, 1, 2)

            ## Sample Spatially
            sampled_video = get_spatial_fragments(video, **sample_args)
            mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])
            sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
            
            sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1, *sampled_video.shape[2:]).transpose(0,1)
            vsamples[sample_type] = sampled_video.to(args.device)
            print(sampled_video.shape)
        result = evaluator(vsamples)
        score = sigmoid_rescale(result.mean().item(), model=args.model)
        print(f"The quality score of the video (range [0,1]) is {score:.5f}.")
        
        ## put score into dictionary
        video_title = video_name.split(ignore_split)[0]
        d_score[video_title] = score
    
    ## output to csv
    if col_video != []:
        with open(args.output_path, 'w') as f:
            f.writelines('video,Prediction\n')
            if col_video == []:
                col_video = list(d_score.keys())
            for video_title in col_video:
                line = video_title + ',' + str(d_score[video_title]) + '\n'
                f.writelines(line)
    else:
        with open('./output.txt', 'w') as f:
            col_video = list(d_score.keys())
            for video_title in col_video:
                f.write(d_score[video_title])