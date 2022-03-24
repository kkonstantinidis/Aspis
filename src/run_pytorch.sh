use_hostfile=true
# use_hostfile=false

# ~ this is the main file to do everything after EC2 setup, do not use pytorch_ec2.py for tasks that are done here, it has dead code
PROJECT_INDEX="$1"
n=21
group_size=3
bucket_size=10
batch_size=50000
rama_m=5
eval_freq=1
max_steps=16
lr_step=$((2*${eval_freq}))
# lr_step=34
checkpoint_step=0

# approach=baseline
# approach=maj_vote
# approach=cyclic
# approach=draco_lite
# approach=draco_lite_attack
# approach=mols
# approach=rama_one
# approach=rama_two
# approach=subset
approach=cyclic_c3les

err_mode=rev_grad
# err_mode=constant
# err_mode=foe

# err_choice=all
err_choice=fixed_disagreement

lis_simulation=simulate
# lis_simulation=nope

mode=coord-median
# mode=bulyan
# mode=multi-krum
# mode=sign-sgd
# mode=maj_vote

adversarial_detection=nope
# adversarial_detection=clique

byzantine_gen=hard_coded
# byzantine_gen=random

hostfile="tools/hosts_address"
# hostfile="tools/hosts_address_local"

local_remote=remote
# local_remote=local

detox_attack=worst
# detox_attack=benign
# detox_attack=whole_group

# lr_warmup=yes
lr_warmup=no

maxlr=0.4

maxlr_steps=9

lr_annealing=yes
# lr_annealing=no

lr_annealing_minlr=0.1

lr_annealing_cycle_steps=66

max_grad_l2norm=0.25

no_cuda=yes
# no_cuda=no

cyclic_ell=2

# ~ test
# for tuning with varying q
tune_dir=${HOME}/shared/tune/ASPIS${PROJECT_INDEX}
echo "Start parameter tuning ..."
for q in 2
do
    for lr in 0.1
    do
        for gamma in 0.95
        do
            START=$(date +%s.%N)

            echo "Trial running for q: ${q}"
            mkdir -p "${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}"
            params=()
            [[ $use_hostfile == true ]] && params+=( '--hostfile' "../${hostfile}" )
            echo "MPI hostfile: ${params[@]}"
            mpirun -n ${n} "${params[@]}" \
            python distributed_nn.py \
            --lr=${lr} \
            --momentum=0.9 \
            --network=ResNet18 \
            --dataset=Cifar10 \
            --batch-size=${batch_size} \
            --comm-type=Bcast \
            --mode=${mode} \
            --approach=${approach} \
            --eval-freq=${eval_freq} \
            --err-mode=${err_mode} \
            --adversarial=-100 \
            --epochs=50 \
            --max-steps=${max_steps} \
            --worker-fail=${q} \
            --group-size=${group_size} \
            --compress-grad=compress \
            --bucket-size=${bucket_size} \
            --checkpoint-step=${checkpoint_step} \
            --lis-simulation=${lis_simulation} \
            --train-dir="${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}/" \
            --local-remote=${local_remote} \
            --rama-m=${rama_m} \
            --byzantine-gen=${byzantine_gen} \
            --detox-attack=${detox_attack} \
            --gamma=${gamma} \
            --lr-step=${lr_step} \
            --err-choice=${err_choice} \
            --adversarial-detection=${adversarial_detection} \
            --lr-warmup=${lr_warmup} \
            --maxlr=${maxlr} \
            --maxlr-steps=${maxlr_steps} \
            --lr-annealing=${lr_annealing} \
            --lr-annealing-minlr=${lr_annealing_minlr} \
            --lr-annealing-cycle-steps=${lr_annealing_cycle_steps} \
            --max-grad-l2norm=${max_grad_l2norm} \
            --no-cuda=${no_cuda} \
            --cyclic-ell=${cyclic_ell}

            END=$(date +%s.%N)
            DIFF=$(echo "$END - $START" | bc)
            echo "Total training time (sec): $DIFF"
        done
    done
done


# Method 1 (from previous versions of ASPIS)
# tune_dir=${HOME}/shared/tune/ASPIS${PROJECT_INDEX}
# echo "Start parameter tuning ..."
# for q in 0 2 6
# do
    # for lr in 0.1
    # do
        # for gamma in 0.975
        # do
            # START=$(date +%s.%N)

            # echo "Trial running for q: ${q}"
            # mkdir -p "${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}"
            # mpirun -n ${n} --hostfile ../${hostfile} \
            # python distributed_nn.py \
            # --lr=${lr} \
            # --momentum=0.9 \
            # --network=LeNet \
            # --dataset=MNIST \
            # --batch-size=${batch_size} \
            # --comm-type=Bcast \
            # --mode=${mode} \
            # --approach=${approach} \
            # --eval-freq=${eval_freq} \
            # --err-mode=${err_mode} \
            # --adversarial=-100 \
            # --epochs=50 \
            # --max-steps=${max_steps} \
            # --worker-fail=${q} \
            # --group-size=${group_size} \
            # --compress-grad=compress \
            # --bucket-size=${bucket_size} \
            # --checkpoint-step=${checkpoint_step} \
            # --lis-simulation=${lis_simulation} \
            # --train-dir="${tune_dir}/output_q_${q}_lr_${lr}_gamma_${gamma}/" \
            # --local-remote=${local_remote} \
            # --rama-m=${rama_m} \
            # --byzantine-gen=${byzantine_gen} \
            # --detox-attack=${detox_attack} \
            # --gamma=${gamma} \
            # --lr-step=${lr_step} \
            # --err-choice=${err_choice} \
            # --adversarial-detection=${adversarial_detection} \
            # --lr-warmup=${lr_warmup} \
            # --maxlr=${maxlr} \
            # --maxlr-steps=${maxlr_steps} \
            # --lr-annealing=${lr_annealing} \
            # --lr-annealing-minlr=${lr_annealing_minlr} \
            # --lr-annealing-cycle-steps=${lr_annealing_cycle_steps} \
            # --max-grad-l2norm=${max_grad_l2norm} \
            # --no-cuda=${no_cuda}

            # END=$(date +%s.%N)
            # DIFF=$(echo "$END - $START" | bc)
            # echo "Total training time (sec): $DIFF"
        # done
    # done
# done

