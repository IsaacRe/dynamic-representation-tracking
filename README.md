# Incremental Object Learning from Contiguous Views
Code for the CVPR 2019 paper [Incremental Object Learning from Contiguous Views](link) 

If you use this code, please cite our paper :
```
Bibitem
```


### Dependencies
For building the color jittering C++ module (Cython required), 
run the following commands:
```
cd utils/color_jitter
python setup.py build_ext --inplace
cd ../..
```

### Running Incremental Learning Models
```
usage: main.py [-h] [--outfile OUTFILE] [--save_all]
               [--save_all_dir SAVE_ALL_DIR] [--resume]
               [--resume_outfile RESUME_OUTFILE] [--init_lr INIT_LR]
               [--init_lr_ft INIT_LR_FT] [--num_epoch NUM_EPOCH]
               [--num_epoch_ft NUM_EPOCH_FT] [--lrd LRD] [--wd WD]
               [--batch_size BATCH_SIZE] [--llr_freq LLR_FREQ]
               [--batch_size_test BATCH_SIZE_TEST] [--lexp_len LEXP_LEN]
               [--size_test SIZE_TEST] [--num_exemplars NUM_EXEMPLARS]
               [--img_size IMG_SIZE] [--rendered_img_size RENDERED_IMG_SIZE]
               [--total_classes TOTAL_CLASSES] [--num_iters NUM_ITERS]
               [--algo ALGO] [--no_dist] [--pt] [--ncm] [--diff_order]
               [--pre_augment] [--no_jitter] [--h_ch H_CH] [--s_ch S_CH]
               [--l_ch L_CH] [--test_freq TEST_FREQ]
               [--num_workers NUM_WORKERS] [--one_gpu]

Incremental learning

optional arguments:
  -h, --help            show this help message and exit
  --outfile OUTFILE     Output file name (should have .csv extension)
  --save_all            Option to save models after each test_freq number of
                        learning exposures
  --save_all_dir SAVE_ALL_DIR
                        Directory to store all models in
  --resume              Resume training from checkpoint at outfile
  --resume_outfile RESUME_OUTFILE
                        Output file name after resuming
  --init_lr INIT_LR     initial learning rate
  --init_lr_ft INIT_LR_FT
                        Init learning rate for balanced fine tuning (for E2E)
  --num_epoch NUM_EPOCH
                        Number of epochs
  --num_epoch_ft NUM_EPOCH_FT
                        Number of epochs for balanced finetuning (for E2E)
  --lrd LRD             Learning rate decrease factor
  --wd WD               Weight decay for SGD
  --batch_size BATCH_SIZE
                        Mini batch size for training
  --llr_freq LLR_FREQ   Learning rate lowering frequency for SGD (for E2E)
  --batch_size_test BATCH_SIZE_TEST
                        Mini batch size for testing
  --lexp_len LEXP_LEN   Number of frames in Learning Exposure
  --size_test SIZE_TEST
                        Number of test images per object
  --num_exemplars NUM_EXEMPLARS
                        number of exemplars
  --img_size IMG_SIZE   Size of images input to the network
  --rendered_img_size RENDERED_IMG_SIZE
                        Size of rendered images
  --total_classes TOTAL_CLASSES
                        Total number of classes
  --num_iters NUM_ITERS
                        Number of learning exposures
  --algo ALGO           Algorithm to run. Options : icarl, e2e, lwf
  --no_dist             Option to switch off distillation loss
  --pt                  Option to start from an ImageNet pretrained model
  --ncm                 Use nearest class mean classification (for E2E)
  --diff_order          Use a random order of classes introduced
  --no_jitter           Option for no color jittering (for iCaRL)
  --h_ch H_CH           Color jittering : max hue change
  --s_ch S_CH           Color jittering : max saturation change
  --l_ch L_CH           Color jittering : max lightness change
  --test_freq TEST_FREQ
                        Number of iterations of training after which a test is
                        done/model saved
  --num_workers NUM_WORKERS
                        Maximum number of threads spawned at any stage of
                        execution
  --one_gpu             Option to run multiprocessing on 1 GPU

```

