max_seq_lens=(256 512 1024 2048 2560 3072 4096)

for max_seq_len in "${max_seq_lens[@]}"
do
    echo "Running with max_seq_len = $max_seq_len"
    nsys profile \
    -o profile-llama-2-7b-chat-max-seq-$max_seq_len \
    --force-overwrite true \
    torchrun --nproc_per_node 1 copy_test.py \
    --ckpt_dir /projects/bcby/qlao/llama-2-7b-chat \
    --tokenizer_path /projects/bcby/qlao/tokenizer.model \
    --max_seq_len $max_seq_len --max_batch_size 1
done