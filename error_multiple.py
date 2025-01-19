import os
import sys
import argparse
from calc_error import calc_error
import glob
import json

def main(gen_path, gt_path, project_dir, json_path, frames = None):
    if not os.path.exists(gen_path):
        print('generated videos file not exists')
        return
    if not os.path.exists(gt_path):
        print('ground truth videos file not exists')
        returnd
    with open(json_path, 'r') as f:
        metadata = json.load(f)
    avg_RotErr = 0
    avg_TransErr = 0
    num_videos = len(metadata)
    for data in metadata:
        video_gen = os.path.join(gen_path, data['video_file_path'])
        video_gt = os.path.join(gt_path, data['video_file_path'])
        project_gen = os.path.join(project_dir, data['video_file_path'])
        project_gt = os.path.join(project_dir, data['video_file_path'])
        roterr, transerr = calc_error(video_gen, video_gt, project_gen, project_gt, frames)
        avg_RotErr += roterr
        avg_TransErr += transerr
    avg_RotErr /= num_videos
    avg_TransErr /= num_videos
    print(f'average rotation error: {avg_RotErr}')
    print(f'average translation error: {avg_TransErr}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type = str)
    parser.add_argument('--gt_path', type = str, default = '/home/lingcheng/RealEstate10KAfterProcess')
    parser.add_argument('--json_path', type = str, default = '/home/lingcheng/RealEstate10KAfterProcess/medadata.json')
    parser.add_argument('--project_path', type = str)
    args = parser.parse_args()
    main(args.gen_path, args.gt_path, args.project_path, args.json_path)