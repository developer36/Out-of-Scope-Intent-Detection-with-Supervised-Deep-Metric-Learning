
for dataset in 'banking'
do
    for known_cls_ratio in 0.5
    do
        for labeled_ratio in 1.0
        do
            for seed in 5 6 7
            do
                for epsilon in 0.5
                do
                    python run.py \
                    --data_dir './dataset/' \
                    --dataset $dataset \
                    --epsilon $epsilon \
                    --method 'ADB' \
                    --known_cls_ratio $known_cls_ratio \
                    --labeled_ratio $labeled_ratio \
                    --seed $seed \
                    --backbone 'bert' \
                    --config_file_name 'ADB' \
                    --loss_fct 'CrossEntropyLoss' \
                    --gpu_id '0' \
                    --pretrain \
                    --train \
                    --results_file_name 'results_ADB_atis_FGM.csv' \
                    --save_results
                done
            done
        done
    done
done