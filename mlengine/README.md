**All scripts assume you are in the mlengine folder.**
## Train locally
```bash
python trainer/task.py
```
or
```bash
mlengine mgorner$ gcloud ml-engine local train --module-name trainer.task --package-path trainer
```
## Train in the cloud
(jobXXX, jobs/jobXXX, &lt;<project&gt; and &lt;bucket&gt; must be replaced with your own values)
```bash
gcloud ml-engine jobs submit training jobXXX --job-dir gs://<bucket>/jobs/jobXXX --project <project> --config=config.yaml --module-name trainer.task --package-path trainer
```
## Predictions from the cloud
Use the Cloud ML Engine UI to create a model and a version from
the saved data from your training run.
You will find it in folder:

gs://&lt;bucket&gt;/jobs/jobXXX/export/Servo/XXXXXXXXXX

Set your version of the model as the default version, then
create the JSON payload. You can use the script:
```bash
python digits.py > digits.json
```
Then call the online predictions service, replacing <model_name> with the name you have assigned:
```bash
gcloud ml-engine predict --model <model_name> --json-instances digits.json
```
It should return a perfect scorecard:

| CLASSES  | PREDICTIONS |
| ------------- | ------------- |
| 8  | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]  |
| 7  | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  |
| 7  | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  |
| 5  | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  |
| 5  | [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]  |
## Local predictions
You can also simulate the prediction service locally, replace XXXXX with the # of your saved model:
```bash
gcloud ml-engine local predict --model-dir checkpoints/export/Servo/XXXXX --json-instances digits.json
```

---
### Misc.
If you want to experiment with TF Records, the sandard Tensorflow
data format, you can run this script ((availble in the tensorflow distribution)
to reformat the MNIST dataset into TF Records. It is not necessary for this sample though.

```bash
python <YOUR-TF-DIR>/tensorflow/examples/how_tos/reading_data/convert_to_records.py --directory=data --validation_size=0
```

