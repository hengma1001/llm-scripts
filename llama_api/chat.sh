torchrun --nproc_per_node 1 --master_port 45600 example_chat_completion.py     --ckpt_dir ../../llama/llama-2-7b-chat/     --tokenizer_path ../../llama/tokenizer.model     --max_seq_len 512 --max_batch_size 6