#!/bin/bash

python /home/deokhk/coursework/robust_speech/recipes/universal_evaluate.py attack_configs/universal/s2s_5000bpe_universal.yaml --data_csv_name=test-clean-adv-100 --root=/home/deokhk/coursework/robust_speech/root --nb_iter=30 --eps=0.05 --lr 0.001 --seed=1026
