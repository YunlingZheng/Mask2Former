python train_net.py --num-gpus 1\
  --config-file configs/coco/instance-segmentation/swin/maskformer2_faster.yaml\
  MODEL.FASTER.PATH_SEARCH /home/macho/Mask2Former/search-224x448_F12.L16_batch5-20200802-144323\
  MODEL.FASTER.PATH_PRETRAIN /home/macho/Mask2Former/train-512x1024_student_batch12-20210613-170252\
  SOLVER.IMS_PER_BATCH 16