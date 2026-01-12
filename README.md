This repository provides **a foundational framework** for federated recommendation systems. 
## !! Before running the code, please create a `log` directory in the root directory of the model.
```
.
├── .idea/            
├── data/
           
├── log/
           
├── README.md         
├── data.py           
├── engine.py        
├── metrics.py        
├── mlp.py            
├── train.py         
├── utils.py          
```

## Notes
This repository is **modified from the open-source PFedRec codebase**:
* Original repository: (https://github.com/Zhangcx19/IJCAI-23-PFedRec)
* **`train.py`** is the main training entry script.
* All federated training, client sampling, local updates, and global aggregation are launched from this file.
* The parameters of the **`affine_output` layer** can be **interpreted as user embeddings**. (mlp.py)


You can start training directly via:
```bash
python train.py [arguments]
```

## Example Usage

```bash
python train.py --dataset ml-100k --num_round 100 --local_epoch 10 --batch_size 256 --lr 0.05 --optimizer adam --use_cuda True --device_id 0
```
