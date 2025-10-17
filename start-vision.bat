@echo off
echo WELCOME TO ASTRAMECH VISION CHECKER
echo Vision Checker v1.0 (C) Bayani Elogada
echo Starting up vision script...
cd c:/astramech
python c:/astramech/vision-check.py --latest-file vision-latest.json --topk 2 --print-rate 1 --use-all-models --conf 0.55