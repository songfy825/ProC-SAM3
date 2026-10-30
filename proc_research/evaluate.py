"""The sole ground-truth reader. Inference/calibration never import label files."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .cache import sha256,write_json


def map_labels(labels,dataset):
    y=np.asarray(labels,dtype=np.int64).copy()
    if dataset=='LoveDA':
        ignore=(y==0)|(y==255);y-=1;y[ignore]=255
    elif dataset!='iSAID':raise ValueError('unsupported dataset')
    return y


def confusion_matrix(target,prediction,num_classes,ignore=255):
    target=np.asarray(target);prediction=np.asarray(prediction)
    if target.shape!=prediction.shape:raise ValueError('full-image shape mismatch')
    if not np.issubdtype(prediction.dtype,np.integer):raise ValueError('labels must be integers')
    if ((prediction<0)|(prediction>=num_classes)).any():raise ValueError('invalid prediction labels')
    keep=target!=ignore;t=target[keep];p=prediction[keep]
    if ((t<0)|(t>=num_classes)).any():raise ValueError('invalid GT labels')
    return np.bincount(num_classes*t.astype(np.int64)+p.astype(np.int64),
                       minlength=num_classes**2).reshape(num_classes,num_classes)


def metrics(cm):
    cm=np.asarray(cm,dtype=np.float64);union=cm.sum(0)+cm.sum(1)-np.diag(cm)
    iou=np.divide(np.diag(cm),union,out=np.full(len(cm),np.nan),where=union>0)
    return {'miou':float(np.nanmean(iou)) if np.isfinite(iou).any() else None,
            'iou':[float(x) if np.isfinite(x) else None for x in iou],
            'confusion_matrix':cm.astype(np.int64).tolist()}


def evaluate_predictions(prediction_index,label_manifest,output):
    """Label manifest is separate: dataset/image_id/path/sha256 only."""
    index=json.loads(Path(prediction_index).read_text())
    labels=[json.loads(x) for x in Path(label_manifest).read_text().splitlines() if x.strip()]
    lookup={(x['dataset'],x['image_id']):x for x in labels}
    if len(lookup)!=len(labels):raise ValueError('duplicate label keys')
    p=Path(output);p.mkdir(parents=True,exist_ok=False)
    grouped={};per_image=[];scenes={}
    for r in index['predictions']:
        if r['split']=='calibration':raise ValueError('calibration GT evaluation forbidden')
        label=lookup[(r['dataset'],r['image_id'])]
        if sha256(label['path'])!=label['sha256'] or sha256(r['path'])!=r['sha256']:
            raise ValueError('prediction/GT integrity mismatch')
        target=map_labels(np.asarray(Image.open(label['path'])),r['dataset'])
        pred=np.load(r['path'],allow_pickle=False)
        cm=confusion_matrix(target,pred,r['num_classes'])
        key=(r['dataset'],r['method']);scene=(*key,r['scene_id'])
        grouped[key]=grouped.get(key,np.zeros_like(cm))+cm
        scenes[scene]=scenes.get(scene,np.zeros_like(cm))+cm
        per_image.append({'dataset':r['dataset'],'image_id':r['image_id'],'scene_id':r['scene_id'],
                          'method':r['method'],**metrics(cm)})
    results={f'{d}/{m}':metrics(cm) for (d,m),cm in grouped.items()}
    write_json(p/'metrics.json',results)
    write_json(p/'per_image.json',per_image)
    write_json(p/'per_scene.json',[dict(dataset=d,method=m,scene_id=s,confusion_matrix=cm.tolist()) for (d,m,s),cm in scenes.items()])
    with open(p/'per_class.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','method','class_id','iou'])
        for (d,m),cm in grouped.items():
            for c,iou in enumerate(metrics(cm)['iou']):writer.writerow([d,m,c,iou])
    with open(p/'per_scene.csv','w',newline='') as f:
        writer=csv.writer(f);writer.writerow(['dataset','method','scene_id','miou'])
        for (d,m,s),cm in sorted(scenes.items()):writer.writerow([d,m,s,metrics(cm)['miou']])
    write_json(p/'provenance.json',{'prediction_index_sha256':sha256(prediction_index),
        'label_manifest_sha256':sha256(label_manifest),
        'output_sha256':{x.name:sha256(x) for x in p.iterdir() if x.is_file()}})
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions',required=True);parser.add_argument('--labels',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args()
    evaluate_predictions(args.predictions,args.labels,args.output)


if __name__=='__main__':main()
