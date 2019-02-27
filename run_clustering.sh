source ./pythonpath.sh

#Name of the files in the HDFS
INPUT_FILE="cracks.csv"
OUTPUT_FILE="clustering_result"
#Size of the block
BLOCK_SIZE="10000"
T_WINDOW="200"
V_WINDOW="0.25"

#Python executor
PYTHON=python36

#Main script
MAIN=main.py
$PYTHON $MAIN $INPUT_FILE $OUTPUT_FILE $BLOCK_SIZE $T_WINDOW $V_WINDOW
