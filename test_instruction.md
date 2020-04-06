This is a brief instruction to perform our test

1. instruction on train.py
```ruby
python train.py 
	--use_validation 
	--dataset_path 		/path/to/megadepth 
	--scene_info_path 	/path/to/preprocessing/output
	--model_type 		vgg16 or res50 or res101
	--truncated_block 	1 or 2 or 3 (resnet usually starts from 2 since it downscale too much)
	--finetune_layers	1 or more (for vgg is the layers, for resnet is bottlenecks)
	--finetune_skip_layers  True or False
	--dilation_blocks	1 or 2
```
After training, pls save the checkpoint and log.txt for evaluation (pls conform a name convention as below)
```ruby
tar -czvf modelType_TruncatedBlock_FinetuneLayers_FinetuneSkipLayers.tar.gz checkpoints/ log.txt
```
2. instruction on extract_feature.py
```ruby
python extract_features.py 
	--image_list_file	/path/to/test_image.txt 
	--model_type 		vgg16 or res50 or res101
	--truncated_blocks 	have to use the same number of training
	--dilation_blocks	1 or 2
```

The output file is installed in the image folder
