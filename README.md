Advanced MLflow and Data Versioning with DVC
Student: Abhinav A Mishra  
Roll number: 24AI005  
Project: MLOps Lab Assignment 4--5  
Repository:
https://github.com/abhinavmishragsv-analytics/MLOps-Lab-3-4
Overview
This project extends a reproducible machine-learning workflow with
MLflow experiment tracking and model registration, and DVC dataset
versioning. The dataset is the scikit-learn Breast Cancer Wisconsin
dataset exported as `data/dataset.csv` (569 rows, 30 features and one
target column).
The workflow records model hyperparameters, evaluation metrics, model
artifacts and input/output signature information. Git commit and DVC
dataset hash are recorded as MLflow tags so that a run can be linked to
the code and data used to create it.
Environment
Windows / PowerShell
Python 3.13.14
MLflow 3.15.2
scikit-learn 1.9.0
DVC 3.67.1
pandas, NumPy, matplotlib and PyYAML (as required by the project)
Install dependencies in the project's virtual environment:
``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```
Start MLflow
From the repository root:
``` powershell
.\.venv\Scripts\mlflow.exe ui --backend-store-uri "sqlite:///C:/Users/Lenovo/Documents/mlops-lab-3-4/mlflow.db" --host 127.0.0.1 --port 5000
```
Open http://127.0.0.1:5000 in a browser. Keep the terminal running
while using the UI.
> The tracking URI above is specific to the original Windows project
> location. If the repository is cloned elsewhere, update the absolute
> path or configure a relative/portable tracking location.
Train candidate variants
The three configuration files are:
`configs/config_e1.yaml`
`configs/config_e2.yaml`
`configs/config_e3.yaml`
Run the variants from the repository root:
``` powershell
python .\src\train.py --config .\configs\config_e1.yaml
python .\src\train.py --config .\configs\config_e2.yaml
python .\src\train.py --config .\configs\config_e3.yaml
```
Use the argument form supported by the current `src/train.py`; if the
script is configured to read a default configuration instead, use the
command documented in that script/your course handout. MLflow runs log
parameters, metrics, model artifacts, evaluation reports, and provenance
tags.
Recorded candidate runs
---
Variant   MLflow run ID                            Trees Max depth   Accuracy   Precision    Recall        F1
---
E1        `2498fc2938904223b079499ea75db7e4`          50         3     0.9649      0.9589    0.9859    0.9722
E2        `6af7d06f2f1a434481a85a48112cf5c5`          50         5     0.9649      0.9589    0.9859    0.9722
E3        `763e991dfdaa49e2bb1d4d120fe3840d`         100         5     0.9649      0.9589    0.9859    0.9722
The precision, recall and F1 above use class `1` (benign) as the
positive class. The confusion matrix, using labels `[0, 1]`, is
`[[40, 3], [1, 70]]`. All three candidate variants had the same recorded
test metrics, so E3 should not be described as more accurate. Version 5
was selected for the production/champion alias as a lifecycle
demonstration, using the larger ensemble as a model-configuration choice
rather than a measured accuracy improvement.
Model signature and artifacts
Models are logged with an MLflow signature and a representative input
example. The signature describes the expected 30-feature input schema
and the single prediction output. Logged artifacts include a confusion
matrix and classification report; the autologging run additionally
records plots such as the training confusion matrix, ROC curve, and
precision-recall curve.
A signature helps identify malformed inputs such as missing columns or
unexpected types, but it does not guarantee that inputs are clinically
appropriate, that predictions are correct, or that a model is safe for
deployment.
Autologging
The autologging demonstration uses the scikit-learn MLflow integration.
The run `0877e9f0364245cdb7faf933ec253172` recorded test accuracy of
approximately `0.9649` and training metrics/plots. Autologging is
flavor-dependent; inspect the run rather than assuming every desired
parameter or artifact is captured automatically.
Model Registry and promotion
Registered model name: `BreastCancerRandomForest`
Registered version   Variant       Lifecycle tag   Alias
---
3                    `config_e1`   `retired`       None
4                    `config_e2`   `candidate`     `candidate`
5                    `config_e3`   `production`    `production`, `champion`
MLflow 3.15.2's new registry interface supports aliases and tags. The
lifecycle state is represented with tags; the alias identifies an
intended use. `retired` is a tag, not an alias. The candidate,
production and retired states are a documented workflow, not a claim of
clinical validation.
DVC dataset versioning
The dataset is tracked with `data/dataset.csv.dvc`; the CSV content is
not tracked as a normal Git file. The `.dvc` metadata records a content
hash and file size, and Git history tracks changes to this metadata.
---
Version      Git commit               Rows        File size DVC MD5
---
v1 baseline  `274a2e5`                 569    121,955 bytes `24d9881fad8cfb3d94f5276d372e422b`
v2 demo      `604ca5a`                 570    122,168 bytes `bc6bbbe19493767b7850f219e3081b09`
Version 2 was created by appending a duplicate of the first row. The
baseline was restored through the commit `ff04782` and `dvc checkout`.
Verification confirmed 569 rows and 31 columns. The subsequent
`dvc status` output was `Data and pipelines are up to date.`
The configured remote is a local directory,
`C:\Users\Lenovo\Documents\dvc-storage`. It is suitable for
demonstrating push and recovery on the same machine, but it is not a
cloud/shared remote. For collaboration, configure an approved shared
storage remote and secure credentials separately.
Useful verification commands:
``` powershell
.\.venv\Scripts\dvc.exe status
.\.venv\Scripts\dvc.exe remote list
git log -8 --oneline
python -c "import pandas as pd; d=pd.read_csv('data/dataset.csv'); print('Rows:',len(d)); print('Columns:',len(d.columns))"
```
Git, DVC and MLflow together
Git records source code, configuration files, requirements,
reports, and DVC metadata.
DVC identifies dataset contents and restores the data version
referenced by a Git revision.
MLflow records experiment parameters, metrics, artifacts, model
signatures, run metadata, and registered model versions.
To investigate a prediction, a teammate can identify the production
alias/version in MLflow, inspect the source run and its Git/DVC tags,
check out the corresponding Git revision, restore the matching dataset
using DVC, and inspect the logged model/artifacts. The teammate should
verify dependency versions and input schema before reproducing
predictions.
DVC pipeline design (proposed extension)
A two-stage DVC pipeline could use:
`prepare`: input `data/dataset.csv` and preprocessing code; output a
reproducible train/test dataset or split metadata.
`train`: prepared data plus one of the YAML configs and
`src/train.py`; outputs the trained model and evaluation report.
The training stage can start an MLflow run and log the Git SHA, DVC data
hash, config, metrics, model signature and artifacts. This is a pipeline
design proposal; the current project has verified DVC dataset tracking
and MLflow training, but a committed `dvc.yaml` pipeline has not been
verified as implemented.
Known limitations and challenges
The DVC remote is local rather than shared/cloud-hosted.
The three variants achieved identical test metrics; E3 is not
demonstrably more accurate.
The breast-cancer dataset is a teaching dataset. The model and
thresholds are not clinically validated and must not be used for
medical decisions.
The MLflow tracking URI in the launch command contains an absolute
Windows path and needs adjustment on another machine.
MLflow may display a deprecation warning for the `artifact_path`
argument in model logging; this warning did not prevent the runs
from completing.
Evidence and report
The accompanying `Report.pdf` includes the project method, results,
registry and DVC discussion, plus the screenshots available from the
MLflow UI. Additional individual screenshot files are in `screenshots/`.
Repository
https://github.com/abhinavmishragsv-analytics/MLOps-Lab-3-4
