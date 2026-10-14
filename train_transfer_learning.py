# -*- coding: utf-8 -*-
"""CIFAR-10 bird-cat-dog transfer learning. Online data and ImageNet weights required."""

# Для Colab: как правило, эти пакеты уже установлены. При необходимости выполните:
# %pip install -q tensorflow scikit-learn matplotlib pillow seaborn
import os, json, random, time, shutil, zipfile, platform
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score, f1_score,
                             ConfusionMatrixDisplay)

SEED = 42
np.random.seed(SEED); random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass
ROOT = Path('/content/transfer_learning_lab') if Path('/content').exists() else Path.cwd() / 'transfer_learning_lab'
ART = ROOT / 'artifacts'; ART.mkdir(parents=True, exist_ok=True)
print('Python:',platform.python_version(),'TensorFlow:',tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
print('Каталог результатов:', ART)

(x_full,y_full),(x_test_official,y_test_official)=tf.keras.datasets.cifar10.load_data()
y_full=y_full.reshape(-1); y_test_official=y_test_official.reshape(-1)
CIFAR_IDS = [2,3,5]
NAMES = ['bird','cat','dog']
map_idx={cid:i for i,cid in enumerate(CIFAR_IDS)}
small_indices=np.concatenate([np.random.default_rng(SEED+cid).choice(np.flatnonzero(y_full==cid),100,replace=False) for cid in CIFAR_IDS])
xs=x_full[small_indices]; ys=np.array([map_idx[int(y)] for y in y_full[small_indices]],dtype=np.int32)
small_tr,small_te=train_test_split(np.arange(len(ys)),test_size=0.2,random_state=SEED,stratify=ys)
SMALL=ROOT/'dataset_100_per_class'
for portion,indices in [('train',small_tr),('test',small_te)]:
    for cls,cn in enumerate(NAMES):
        dest=SMALL/portion/cn; dest.mkdir(parents=True,exist_ok=True)
        selected=indices[ys[indices]==cls]
        for n,idx in enumerate(selected):
            Image.fromarray(xs[idx]).resize((224,224),Image.Resampling.LANCZOS).save(dest/f'{cn}_{n:03d}.png')
        print(portion,cn,len(selected))
ZIPDATA=ART/'dataset_100_per_class.zip'
with zipfile.ZipFile(ZIPDATA,'w',zipfile.ZIP_DEFLATED) as z:
    for file in sorted(SMALL.rglob('*.png')): z.write(file,file.relative_to(ROOT))
print('Создан:',ZIPDATA,'— классы без пересечений, разделение 80/20')
fig,axes=plt.subplots(3,6,figsize=(12,6))
for i,cn in enumerate(NAMES):
    picks=np.flatnonzero(ys==i)[:6]
    for j,k in enumerate(picks): axes[i,j].imshow(xs[k]);axes[i,j].axis('off');axes[i,j].set_title(cn)
plt.suptitle('CIFAR-10: примеры трех классов');plt.tight_layout()
plt.savefig(ART/'dataset_samples.png',dpi=170,bbox_inches='tight');plt.show()

SIZE=(96,96)
def make_ds(x,y,training=False,batch=32):
    ds=tf.data.Dataset.from_tensor_slices((x.astype('uint8'),np.asarray(y,dtype='int32')))
    if training: ds=ds.shuffle(len(y),seed=SEED,reshuffle_each_iteration=True)
    ds=ds.batch(batch).map(lambda a,b:(tf.image.resize(tf.cast(a,tf.float32),SIZE),b),num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)

def make_model():
    # Backbone заранее обучен на ImageNet — transfer learning, не обучение с нуля
    backbone=tf.keras.applications.MobileNetV2(include_top=False,weights='imagenet',input_shape=(96,96,3))
    backbone.trainable=False
    inp=tf.keras.Input(shape=(96,96,3))
    x=tf.keras.Sequential([tf.keras.layers.RandomFlip('horizontal'),
                           tf.keras.layers.RandomRotation(0.04),
                           tf.keras.layers.RandomZoom(0.08)],name='augmentation')(inp)
    x=tf.keras.layers.Rescaling(1./127.5,offset=-1)(x)
    x=backbone(x,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dense(128,activation='relu')(x)
    x=tf.keras.layers.Dropout(0.3)(x)
    out=tf.keras.layers.Dense(3,activation='softmax')(x)
    model=tf.keras.Model(inp,out)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),metrics=['accuracy'])
    return model

def plot_learning(history,prefix):
    fig,ax=plt.subplots(1,2,figsize=(12,4))
    for key,name in [('accuracy','accuracy'),('loss','loss')]:
        ax[0 if key=='accuracy' else 1].plot(history.history[key],label='train')
        ax[0 if key=='accuracy' else 1].plot(history.history['val_'+key],label='validation')
        ax[0 if key=='accuracy' else 1].set_title(name);ax[0 if key=='accuracy' else 1].set_xlabel('Epoch');ax[0 if key=='accuracy' else 1].legend()
    fig.tight_layout(); fig.savefig(ART/f'{prefix}_learning.png',dpi=180);plt.show()
    plt.close(fig)

inner_tr,inner_val=train_test_split(small_tr,test_size=0.25,stratify=ys[small_tr],random_state=SEED)
model_tm=make_model()
callbacks_tm=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=4,restore_best_weights=True)]
start=time.perf_counter()
h_tm=model_tm.fit(make_ds(xs[inner_tr],ys[inner_tr],True,16),validation_data=make_ds(xs[inner_val],ys[inner_val],False,16),
                  epochs=25,callbacks=callbacks_tm,verbose=2)
time_tm=time.perf_counter()-start
small_result=model_tm.evaluate(make_ds(xs[small_te],ys[small_te],False,16),verbose=0,return_dict=True)
model_tm.save(ART/'small_dataset_transfer.keras')
plot_learning(h_tm,'small_dataset')
print('small external test:', small_result, 'seconds:',round(time_tm,1))

def sample_per_class(x,y,ids,limit):
    rng=np.random.default_rng(SEED)
    ids_all=np.concatenate([rng.choice(np.flatnonzero(y==cid),limit,replace=False) for cid in ids])
    rng.shuffle(ids_all)
    return x[ids_all],np.asarray([map_idx[int(q)] for q in y[ids_all]],dtype=np.int32)
x_sel,y_sel=sample_per_class(x_full,y_full,CIFAR_IDS,1500)
x_train,x_val,y_train,y_val=train_test_split(x_sel,y_sel,test_size=0.2,random_state=SEED,stratify=y_sel)
mask=np.isin(y_test_official,CIFAR_IDS)
x_test=x_test_official[mask]; y_test=np.asarray([map_idx[int(v)] for v in y_test_official[mask]],dtype=np.int32)
print('train',len(y_train),'validation',len(y_val),'test',len(y_test))
model=make_model()
model.summary()
start=time.perf_counter()
hist=model.fit(make_ds(x_train,y_train,True,32),validation_data=make_ds(x_val,y_val,False,32),epochs=8,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=3,restore_best_weights=True),
                          tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',factor=.5,patience=2,min_lr=1e-5)],verbose=2)
train_seconds=time.perf_counter()-start
res_tr=model.evaluate(make_ds(x_train,y_train),verbose=0,return_dict=True)
res_val=model.evaluate(make_ds(x_val,y_val),verbose=0,return_dict=True)
res_test=model.evaluate(make_ds(x_test,y_test),verbose=0,return_dict=True)
probs=model.predict(make_ds(x_test,y_test),verbose=0);pred=probs.argmax(1)
report=classification_report(y_test,pred,target_names=NAMES,digits=4,output_dict=True,zero_division=0)
report_txt=classification_report(y_test,pred,target_names=NAMES,digits=4,zero_division=0)
print(report_txt)
plot_learning(hist,'main')
cm=confusion_matrix(y_test,pred,labels=[0,1,2])
fig,ax=plt.subplots(figsize=(6,5));ConfusionMatrixDisplay(cm,display_labels=NAMES).plot(ax=ax,cmap='Blues',values_format='d',colorbar=False)
ax.set_title('Confusion matrix — CIFAR-10 test, 3 класса');fig.tight_layout()
fig.savefig(ART/'confusion_matrix.png',dpi=180);plt.show();plt.close(fig)
wrong=np.flatnonzero(pred!=y_test)
if len(wrong)>0:
    fig,axes=plt.subplots(2,4,figsize=(11,6))
    for ax,idx in zip(axes.flat,wrong[:8]):
        ax.imshow(x_test[idx]);ax.axis('off');ax.set_title(f'{NAMES[y_test[idx]]} → {NAMES[pred[idx]]}',fontsize=10)
    for ax in list(axes.flat)[min(8,len(wrong)):]:ax.axis('off')
    fig.suptitle('Примеры ошибок');fig.tight_layout();fig.savefig(ART/'misclassifications.png',dpi=160);plt.show();plt.close(fig)
model.save(ART/'mobilenetv2_cifar3.keras')
model.save(ART/'mobilenetv2_cifar3.h5')
print('Модель сохранена в Keras и H5')

recall_by_class=cm.diagonal()/cm.sum(axis=1)
strong_i=int(np.argmax(recall_by_class));weak_i=int(np.argmin(recall_by_class))
errors=[{'true':NAMES[int(y_test[i])],'predicted':NAMES[int(pred[i])]} for i in wrong[:10]]
metrics={
  'date':'2026-10-14','source':'CIFAR-10 / official train and test',
  'architecture':'MobileNetV2 ImageNet frozen + GAP + Dense(128) + Dropout(0.3) + softmax(3)',
  'labels':NAMES,'split':{'training':len(y_train),'validation':len(y_val),'test':len(y_test),
                          'small_dataset_train':len(small_tr),'small_dataset_test':len(small_te)},
  'parameters':{'epochs_max':8,'epochs_executed':len(hist.history['loss']),'batch_size':32,'initial_learning_rate':0.001,
                'validation_split':0.20,'optimizer':'Adam','dropout':0.3},
  'training_time_seconds':round(train_seconds,2),
  'train_accuracy':float(res_tr['accuracy']),'validation_accuracy':float(res_val['accuracy']),
  'train_loss':float(res_tr['loss']),'validation_loss':float(res_val['loss']),
  'test_accuracy':accuracy_score(y_test,pred),
  'test_precision_macro':precision_score(y_test,pred,average='macro',zero_division=0),
  'test_recall_macro':recall_score(y_test,pred,average='macro',zero_division=0),
  'test_f1_macro':f1_score(y_test,pred,average='macro',zero_division=0),
  'confusion_matrix':cm.tolist(),
  'best_class':{'label':NAMES[strong_i],'recall':float(recall_by_class[strong_i])},
  'worst_class':{'label':NAMES[weak_i],'recall':float(recall_by_class[weak_i])},
  'mistakes':errors,
  'small_model':{'epochs_max':25,'epochs_executed':len(h_tm.history['loss']),'batch_size':16,
                 'learning_rate':0.001,'training_time_seconds':round(time_tm,2),
                 'external_test_accuracy':float(small_result['accuracy']),'external_test_loss':float(small_result['loss'])}
}
(ART/'metrics.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf8')
(ART/'classification_report.txt').write_text(report_txt,encoding='utf8')
(ART/'SUMMARY.md').write_text('# Результаты после фактического обучения\n\n' + '\n'.join([f'- **{k}**: {v}' for k,v in metrics.items() if k in ['train_accuracy','validation_accuracy','test_accuracy','test_precision_macro','test_recall_macro','test_f1_macro','training_time_seconds','best_class','worst_class']]),encoding='utf8')
assert len(y_train)==3600 and len(y_val)==900 and len(y_test)==3000
assert cm.sum()==3000 and np.isfinite(list([metrics[k] for k in ['test_accuracy','test_precision_macro','test_recall_macro','test_f1_macro']])).all()
print(json.dumps(metrics,ensure_ascii=False,indent=2))
ARCHIVE=ROOT/'lab_results.zip'
with zipfile.ZipFile(ARCHIVE,'w',zipfile.ZIP_DEFLATED) as z:
  for p in ART.rglob('*'):
    if p.is_file():z.write(p,p.relative_to(ROOT))
print('ВЫПОЛНЕНО. Архив:',ARCHIVE)
try:
    from google.colab import files
    files.download(str(ARCHIVE))
except ImportError:
    print('Скопируйте архив вручную:',ARCHIVE)