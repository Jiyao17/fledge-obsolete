

source ~/fledge/python/bin/activate


for ((i=0; i<$round_num; i++))
    python ~/server/main.py
    python ~/client/main.py