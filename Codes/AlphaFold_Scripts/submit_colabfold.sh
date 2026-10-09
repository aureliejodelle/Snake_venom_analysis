#!/usr/bin/env bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate colabfold

INPUT_DIR="SnakeVenom.fasta.split"
OUTPUT_DIR="colabfold_output_results"
mkdir -p "$OUTPUT_DIR"

total_files=$(ls -1 "$INPUT_DIR"/SnakeVenom.part_*.fasta | wc -l)
echo "Found $total_files split files to process"

for ((i=1; i<=total_files; i++)); do
    part_number=$(printf "%03d" $i)
    INPUT_FILE="${INPUT_DIR}/SnakeVenom.part_${part_number}.fasta"
    TASK_OUTPUT_DIR="${OUTPUT_DIR}/batch_${part_number}"

    # how many sequences are expected in this batch
    n_seqs=$(grep -c "^>" "$INPUT_FILE")

    # how many are already done in this batch's output dir
    if [ -d "$TASK_OUTPUT_DIR" ]; then
        n_done=$(find "$TASK_OUTPUT_DIR" -maxdepth 1 -name "*.done.txt" | wc -l)
    else
        n_done=0
    fi

    if [ "$n_done" -ge "$n_seqs" ]; then
        echo "Skipping batch_${part_number}: already complete ($n_done/$n_seqs done)"
        continue
    fi

    echo "---------------------"
    echo "Starting job on $INPUT_FILE ($n_done/$n_seqs already done)"
    echo "Time: $(date)"
    echo "---------------------"

    colabfold_batch --num-recycle 3 --model-type auto --use-gpu "$INPUT_FILE" "$TASK_OUTPUT_DIR"

    echo "---------------------"
    echo "Finished job on $INPUT_FILE"
    echo "Time: $(date)"
    echo "---------------------"
done

echo "All jobs completed!"

#chmod +x submit_colabfold.sh
#nohup ./submit_colabfold.sh > colabfold_jobs.log 2>&1 &
