# Remove old checkpoint directories (keep only the latest)
cd /home/ubuntu/projects/news_GRPO/ckpt
ls -lh  # See what's there
# rm -rf grpo-Dec260139  # Remove old ones if not needed

# Or free up cache
rm -rf ~/.cache/huggingface/hub/*