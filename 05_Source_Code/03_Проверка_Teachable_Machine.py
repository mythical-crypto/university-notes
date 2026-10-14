#!/usr/bin/env python3
"""Evaluate the PUBLIC Google Teachable Machine model on previously held-out photos.

The 60 photographs must come only from dataset_highres/test, never train.
This is a separate experiment from the previously trained CIFAR-10 Keras model.
"""
import base64
import csv
import json
import zipfile
from collections import Counter
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix,
    classification_report, ConfusionMatrixDisplay, log_loss
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from playwright.sync_api import sync_playwright

MODEL_URL = "https://teachablemachine.withgoogle.com/models/rPbqmrUgK/"
DATASET = Path("02_Dataset/Датасет_Птицы_Кошки_Собаки_80-20.zip")
LABELS = ["bird", "cat", "dog"]

with zipfile.ZipFile(DATASET) as z:
    paths = sorted(n for n in z.namelist() if "/test/" in n and n.lower().endswith((".jpg",".jpeg",".png")))
    assert len(paths)==60, len(paths)
    for label in LABELS:
        assert sum(("/test/"+label+"/") in name for name in paths)==20
    images = [(name, base64.b64encode(z.read(name)).decode("ascii")) for name in paths]

HTML = """<!doctype html><html><head><meta charset='utf-8'></head><body>Model test
<script src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@1.7.4/dist/tf.min.js'></script>
<script src='https://cdn.jsdelivr.net/npm/@teachablemachine/image@0.8/dist/teachablemachine-image.min.js'></script>
</body></html>"""

with sync_playwright() as playwright:
    browser = playwright.chromium.launch(headless=True, args=[
        "--no-sandbox", "--use-gl=angle", "--use-angle=swiftshader", "--disable-dev-shm-usage"
    ])
    page = browser.new_page(viewport={"width": 1024, "height": 768})
    messages=[]
    page.on("console", lambda m: messages.append(f"console {m.type}: {m.text}"))
    page.on("pageerror", lambda e: messages.append(f"pageerror: {e}"))
    page.set_default_timeout(180000)
    page.set_content(HTML, wait_until="load", timeout=180000)
    print("Loaded libraries", page.evaluate("({tf: typeof tf, tmImage: typeof tmImage})"), flush=True)
    init = page.evaluate("""async ({url}) => {
        await tf.setBackend('cpu'); await tf.ready();
        window.predictor = await tmImage.load(url + 'model.json', url + 'metadata.json');
        return {labels:window.predictor.getClassLabels(), input:window.predictor.getTotalClasses()};
    }""", {"url": MODEL_URL})
    print("Model init:",init,flush=True)
    assert init["labels"] == LABELS, init
    results=[]
    for index, (path,data_b64) in enumerate(images, 1):
        prediction = page.evaluate("""async ({data}) => {
          const img = new Image();
          img.src='data:image/jpeg;base64,' + data;
          await img.decode();
          const res=await window.predictor.predict(img, false);
          return res.map(x=>({className:x.className,probability:x.probability}));
        }""", {"data":data_b64})
        assert len(prediction)==3 and [p["className"] for p in prediction]==LABELS, prediction
        probs=[float(p["probability"]) for p in prediction]
        assert all(0<=p<=1.001 for p in probs),probs
        predicted=LABELS[max(range(3),key=lambda i:probs[i])]
        true=path.split("/test/")[1].split("/")[0]
        results.append({"file":path, "true":true,"predicted":predicted,
                        "p_bird":probs[0],"p_cat":probs[1],"p_dog":probs[2]})
        if index%10==0:print(f"Evaluated {index}/60 images",flush=True)
    browser.close()
    print("Browser console tail:",messages[-8:],flush=True)

true=[r["true"] for r in results]
pred=[r["predicted"] for r in results]
scores=[[r["p_"+label] for label in LABELS] for r in results]
pre,rec,f1,sup=precision_recall_fscore_support(true,pred,labels=LABELS,zero_division=0)
macro=precision_recall_fscore_support(true,pred,labels=LABELS,average="macro",zero_division=0)
cm=confusion_matrix(true,pred,labels=LABELS)
out={
  "model_url":MODEL_URL,
  "evaluation_date_utc":datetime.now(timezone.utc).isoformat(),
  "source":"60 unseen high-quality holdout photos, 20 per class, from 02_Dataset/Датасет_Птицы_Кошки_Собаки_80-20.zip",
  "test_images":len(results),"classes":LABELS,"training_per_class":80,"test_per_class":20,
  "test_accuracy":float(accuracy_score(true,pred)),
  "test_precision_macro":float(macro[0]),
  "test_recall_macro":float(macro[1]),
  "test_f1_macro":float(macro[2]),
  "test_log_loss":float(log_loss(true,scores,labels=LABELS)),
  "confusion_matrix":cm.tolist(),
  "per_class":{label:{"precision":float(pre[i]),"recall":float(rec[i]),
      "f1_score":float(f1[i]),"support":int(sup[i])} for i,label in enumerate(LABELS)},
  "classification_report":classification_report(true,pred,labels=LABELS,zero_division=0),
  "important":"Metrics belong to the published Google Teachable Machine model, NOT CIFAR-10 MobileNetV2."
}
dest=Path("03_Teachable_Machine")
(dest/"02_Метрики_60_фото.json").write_text(
  json.dumps(out,ensure_ascii=False,indent=2),encoding="utf-8")
with open(dest/"03_Предсказания_60_фото.csv","w",encoding="utf-8",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["file","true","predicted","p_bird","p_cat","p_dog"])
    w.writeheader();w.writerows(results)
fig,ax=plt.subplots(figsize=(7,6))
ConfusionMatrixDisplay(cm,display_labels=LABELS).plot(ax=ax,cmap="Blues",values_format="d",colorbar=False)
ax.set_title("Teachable Machine — 60 unseen high-quality test photos")
fig.tight_layout();fig.savefig(dest/"04_Матрица_ошибок.png",dpi=170);plt.close(fig)
print(json.dumps({k:v for k,v in out.items() if k.startswith("test_") or k=="confusion_matrix"},indent=2),flush=True)
