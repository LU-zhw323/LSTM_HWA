## LSTM HWA Training & Inference

### Branch

* Main: This branch stored the script for single process task.
* HPC: This branch stored the script for multiple process tasks that run in parallel on HPC.


### Data

The Penn Tree Bank dataset are placed in the ```/data/ptb```

### Pretrained LSTM-FP model

The Pretrained Floating point LSTM model, called ```lstm_fb.pt```, is uploaded to [Hugging Face](https://huggingface.co/MarvinZhw/LSTM-FP-PTB/tree/main). Please download it and place it in the ```/model``` to run the HWA training on it. It had test Perplexity at 85. It can be directly used as a starting point for HWA training and inference. If you prefer to train a new LSTM-FP model, please see the note below.

### Best Condition for LSTM-FP

Just use this cmd:
```python lstm_fp.py --cuda --emsize 650 --nhid 650 --dropout 0.5 --epochs 40```
Once you have it, please put it into the ```/model``` and make sure its name is ```lstm_fp.pt``` which is the default name generated by the script.

### HWA training: lstm_hwa.py

Please check the ```parameters.json``` file and modify it to input the desired parameter for the hwa training. Since the script should be run on HPC through the ```hpc.sh``` script, I created a simple copy of it called ```test.sh``` to simulate the condition of one task on HPC. If you prepare to run the simulation on HPC, please use the ```hpc.sh``` and modify its content, which will create an array of task to run multiple simuations at the same time with different setting, which you can customize your own setting in ```parameters.json```. If you just run it locally or on the server of our lab, please just use the cmd ```test.sh``` with the cmd ```sh test.sh```, which will just simulate one task of hpc. Also Please check the ```set_param()``` at line 60 of the ```lstm_hwa.py``` file for more detail about how to set up parameters.  

The HWA re-trained LSTM, called ```lstm_hwa.th```, is uploaded to [Hugging Face](https://huggingface.co/MarvinZhw/LSTM-HWA-PTB/tree/main). Please download it and place it in the ```/model``` to run the inference on it.  

This script will save the state dictionary of the hwa trained model as ```./model/lstm_hwa.th```. The best condition for the hwa training is saved in the ```./log/lstm_hwa_output.txt```. Notice: For the training noise scale, we run 3 trials which indicated the best noise scale would range from 3.2 to 3.4. We used 3.4 for this experiment.  

### Inference: lstm_inference.py

Save as ```lstm_hwa.py``` which require a shell script and json file with parameters to run on HPC. It will run the inference with the parameters provided on the model.
