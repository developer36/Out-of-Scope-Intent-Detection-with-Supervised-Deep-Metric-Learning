
for dataset in 'banking'
do
    for known_cls_ratio in 0.5
    do
        for labeled_ratio in 1.0
        do
            for seed in 0 1 2
                do
                    python run.py \
                    --pretrain_model_dir '/home/jovyan/oos可视化/my_pretrain' \
                    --data_dir './dataset/' \
                    --dataset $dataset \
                    --epsilon 0.5 \
                    --known_cls_ratio $known_cls_ratio \
                    --labeled_ratio $labeled_ratio \
                    --seed $seed \
                    --loss_fct 'CrossEntropyLoss' \
                    --gpu_id '0' \
                    --pretrain \
                    --train \
                    --results_file_name 'results_banking_FGM.csv' \
                    --save_results
                done
        done
    done
done