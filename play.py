import gym
import pygame
from gym.utils.play import play
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--module", type=str, default="ALE/Breakout-v5")
args = parser.parse_args()

mapping = {(pygame.K_UP,): 1, (pygame.K_RIGHT,): 2, (pygame.K_LEFT,): 3}
play(gym.make(args.module, render_mode="rgb_array"), keys_to_action=mapping)