import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--module", type=str, default="LunarLanderContinuous-v2")
parser.add_argument("-a", "--algorithm", type=str, default="PPO")

args = parser.parse_args()
os.system("ffmpeg -y -i rgb_array/rl-video-episode-0.mp4 -f gif"+" ./images/"+args.module.removeprefix("ALE/")+"_"+args.algorithm+".gif")