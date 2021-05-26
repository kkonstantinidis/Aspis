echo "Starting evaluation ..."

# bash evaluate_pytorch.sh "" 0 0.1 1 50 && \
# bash evaluate_pytorch.sh "" 0 0.1 0.9 50 && \
# bash evaluate_pytorch.sh "" 0 0.1 0.99 50 && \
# bash evaluate_pytorch.sh "" 0 0.1 0.95 50 && \
# bash evaluate_pytorch.sh "" 0 0.01 1 50 && \
# bash evaluate_pytorch.sh "" 0 0.01 0.9 50 && \
# bash evaluate_pytorch.sh "" 0 0.01 0.99 50 && \
# bash evaluate_pytorch.sh "" 0 0.01 0.95 50 && \
# bash evaluate_pytorch.sh "" 0 0.001 1 50 && \
# bash evaluate_pytorch.sh "" 0 0.001 0.9 50 && \
# bash evaluate_pytorch.sh "" 0 0.001 0.99 50 && \
# bash evaluate_pytorch.sh "" 0 0.001 0.95 50

# bash evaluate_pytorch.sh "202" 2 0.1 1 320 && \
# bash evaluate_pytorch.sh "202" 2 0.1 0.95 320 && \
# bash evaluate_pytorch.sh "202" 2 0.1 0.7 320 && \
# bash evaluate_pytorch.sh "202" 2 0.01 1 320 && \
# bash evaluate_pytorch.sh "202" 2 0.01 0.95 320 && \
# bash evaluate_pytorch.sh "202" 2 0.01 0.7 320 && \
# bash evaluate_pytorch.sh "202" 2 0.001 1 320 && \
# bash evaluate_pytorch.sh "202" 2 0.001 0.95 320 && \
# bash evaluate_pytorch.sh "202" 2 0.001 0.7 320

bash evaluate_pytorch.sh "500" 6 0.01 0.7 2000 && \
bash evaluate_pytorch.sh "502" 2 0.01 0.7 2000 && \
bash evaluate_pytorch.sh "502" 4 0.01 0.7 2000 && \
bash evaluate_pytorch.sh "505" 2 0.1 0.7 2000 && \
bash evaluate_pytorch.sh "506" 2 0.01 0.975 2000 && \
bash evaluate_pytorch.sh "506" 4 0.01 0.975 2000 && \
bash evaluate_pytorch.sh "507" 6 0.01 0.7 2000 && \
bash evaluate_pytorch.sh "503" 2 0.01 0.95 2000 && \
bash evaluate_pytorch.sh "503" 4 0.01 0.95 2000

bash evaluate_pytorch.sh "508" 6 0.1 0.95 66 && \
bash evaluate_pytorch.sh "501" 6 0.1 0.95 66 && \
bash evaluate_pytorch.sh "504" 2 0.1 0.95 66 && \
bash evaluate_pytorch.sh "504" 4 0.1 0.95 66



# for PROJECT_INDEX in 1
# do
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.1 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.05 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.025 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0125 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.00625 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.003125 0.96 200
    # bash evaluate_pytorch.sh ${PROJECT_INDEX} 3 0.0015625 0.96 200
# done