from pathlib import Path
import pickle
root='/home/yansu/paper/al_batch_all/run_s3_e30_n10_MoS2'
p=Path(root).glob('**/alstate_1.pkl')

id=[]
for i in p:
    with open(i,'rb') as f:
        state=pickle.load(f)
    data_list=state['dataset']['data_list']
    num_id={i:int(Path(v['img_path']).stem) for i,v in enumerate(data_list)}
    labeled_index_list=state['dataset']['labeled_index_list']
    labeled_index_list=labeled_index_list[1]
    for j in labeled_index_list:
        id.append(num_id[j])
    print(len(labeled_index_list))
print(len(id),len(set(id)))