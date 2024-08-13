#!/bin/sh
python main.py --config configs/church.yml --exp ddim_log --doc church --sample --fid --timesteps 100 --eta 0 --ni
