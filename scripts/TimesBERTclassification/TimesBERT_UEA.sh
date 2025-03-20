export CUDA_VISIBLE_DEVICES=4

model_name=TimesBERT

# 第一个参数为学习率
lr=3e-5
ckpt_path=/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long/08012025_010843/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_long-epoch=02.ckpt
lradj=cosine

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/EthanolConcentration/ \
  --model_id EthanolConcentration \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj
  # --use_finetune

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/FaceDetection/ \
  --model_id FaceDetection \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 8 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/Handwriting/ \
  --model_id Handwriting \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/Heartbeat/ \
  --model_id Heartbeat \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/JapaneseVowels/ \
  --model_id JapaneseVowels \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/PEMS-SF/ \
  --model_id PEMS-SF \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size $batch_size \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
##  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/SelfRegulationSCP1/ \
  --model_id SelfRegulationSCP1 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/SelfRegulationSCP2/ \
  --model_id SelfRegulationSCP2 \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/SpokenArabicDigits/ \
  --model_id SpokenArabicDigits \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
#
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../DATA/UWaveGestureLibrary/ \
  --model_id UWaveGestureLibrary \
  --model $model_name \
  --data UEA \
  --e_layers 3 \
  --batch_size 16 \
  --d_model 128 \
  --d_ff 256 \
  --top_k 3 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate $lr \
  --ckpt_path $ckpt_path \
  --train_epochs 50 \
  --patience 10 \
  --lradj $lradj \
#  --use_finetune
