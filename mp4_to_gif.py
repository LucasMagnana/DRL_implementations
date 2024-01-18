from moviepy.editor import VideoFileClip
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--module", type=str, default="LunarLanderContinuous-v2")
parser.add_argument("-a", "--algorithm", type=str, default="PPO")
parser.add_argument("-s", "--speed", type=int, default=8)

args = parser.parse_args()

videoClip = VideoFileClip("rgb_array/rl-video-episode-0.mp4")
videoClip.speedx(args.speed).write_gif("images/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".gif", fps=120)