cd ~/RevTrack-main

PHASE=all \
POOLS="sum mean" \
SEEDS="0 1 2" \
SETTINGS="10+1000@100 10+10000@100" \
MAX_EPOCHS=300 \
LOG_ROOT=logs-pooling-sensitivity \
bash scripts/run_pooling_sensitivity.sh