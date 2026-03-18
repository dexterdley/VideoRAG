# Dataset for ActivityQA
# huggingface-cli download YimuWang/ActivityNet --repo-type dataset --local-dir ./ActivityNet_Archives --local-dir-use-symlinks False

mkdir -p ./All_Videos

# Extract all tar.gz files
for f in ./ActivityNet_Archives/*.tar.gz; do
    tar -xzf "$f" -C ./All_Videos
done

# Extract all zip files (the 'missing' videos)
for f in ./ActivityNet_Archives/*.zip; do
    unzip -n "$f" -d ./All_Videos
done