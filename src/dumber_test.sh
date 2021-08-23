CUDA_VISIBLE_DEVICES=1 python main.py SpeechCommand 100 9 2000 5 256 0.01 -d cuda -n 5
CUDA_VISIBLE_DEVICES=2 python main.py SpeechCommand 100 3 5000 5 256 0.01 -d cuda        -p ../data -n 5