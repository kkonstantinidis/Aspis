echo "Starting evaluation ..."

bash evaluate_pytorch.sh "" 0 0.1 1 50 && \
bash evaluate_pytorch.sh "" 0 0.1 0.9 50 && \
bash evaluate_pytorch.sh "" 0 0.1 0.99 50 && \
bash evaluate_pytorch.sh "" 0 0.1 0.95 50 && \
bash evaluate_pytorch.sh "" 0 0.01 1 50 && \
bash evaluate_pytorch.sh "" 0 0.01 0.9 50 && \
bash evaluate_pytorch.sh "" 0 0.01 0.99 50 && \
bash evaluate_pytorch.sh "" 0 0.01 0.95 50 && \
bash evaluate_pytorch.sh "" 0 0.001 1 50 && \
bash evaluate_pytorch.sh "" 0 0.001 0.9 50 && \
bash evaluate_pytorch.sh "" 0 0.001 0.99 50 && \
bash evaluate_pytorch.sh "" 0 0.001 0.95 50




# for PROJECT_INDEX in 1
# do
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.1 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.05 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.025 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0125 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.00625 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.003125 0.96 200
    # # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0015625 0.96 200
# done