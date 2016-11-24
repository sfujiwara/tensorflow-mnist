# MNIST Classification with TensorFlow

## Training on Cloud Machine Learning

```
ALGORITHM="cnn"
JOB_NAME="${ALGORITHM}`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_PATH=gs://${PROJECT_ID}-ml/mnist/${JOB_NAME}
gsutil cp .dummy ${TRAIN_PATH}/model/

gcloud beta ml jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=config.yaml \
  -- \
  --output_path="${TRAIN_PATH}" \
  --algorithm="${ALGORITHM}"
```

## Training on Local

```
python -m trainer.task --output_path=log
```