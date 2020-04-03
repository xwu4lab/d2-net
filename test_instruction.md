This is a brief instruction to use the train.py for our test

python train.py 
	--use_validation 
	--dataset_path 		/path/to/megadepth 
	--scene_info_path 	/path/to/preprocessing/output
	--model_type 		vgg16 or res50 or res101
	--truncated_block 	1 or 2 
	--finetune_layers	2 or more

After training, pls save the checkpoint and log.txt for evaluation using

tar -czvf nameOfTestSetting.tar.gz checkpoints/ log.txt
