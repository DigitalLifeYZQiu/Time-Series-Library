export CUDA_VISIBLE_DEVICES=4

#ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full/04012025_193158/ckpt/bert/bert/lotsa_uea_bert_d768_l12_p36_n512_weight_p25_wo_revin_full-epoch=14.ckpt"
ckpt_path="/data/zhanghaoran/projects/ltsm/experiments/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long/20012025_231911/ckpt/bert/bert/lotsa_uea_bert_d256_l4_p4_n512_weight_p25_wo_revin_long-epoch=01.ckpt"

for seq_len in 40
do
for mask_rate in 0
do
echo mask rate $mask_rate
python -u run.py \
  --task_name anomaly_detection \
  --is_training 1 \
  --ckpt_path $ckpt_path \
  --root_path ../DATA/MSL \
  --model_id MSL \
  --model TimesBERT \
  --data MSL \
  --features M \
  --seq_len $seq_len \
  --pred_len 0 \
  --batch_size 64 \
  --patch_len 4 \
  --train_epochs 3 \
  --stride 40 \
  --enc_in 55 \
  --anomaly_ratio 1 \
  --learning_rate 0.0005 \
  --lradj cosine \
  --ad_mask_type random \
  --mask_rate $mask_rate \
  --date_record
done
done