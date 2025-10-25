#%%
from utils import (
    load_run_data,
    collect_micro_to_otus,
    load_microbiome_model,
    preview_prokbert_embeddings,
    build_sample_embeddings,
    get_config,
)

# Load configuration
config = get_config()

#%%
run_rows, SRA_to_micro, gid_to_sample, micro_to_subject, micro_to_sample = load_run_data()
#%%
micro_to_otus = collect_micro_to_otus(SRA_to_micro, micro_to_subject)
#%%
model, device = load_microbiome_model()
#%%
preview_prokbert_embeddings()
#%%
sample_embeddings, missing_otus = build_sample_embeddings(micro_to_otus, model, device)
# %%

"""
From here we can now train some models for prediciton.
For example, lets see if we can predict the milk type.
For each subject we have multiple time points. We can take this into account:
we will do a logitic regression, then visualise our roc-aucs at different time points 
for each category: e.g. mothers_breast_milk, formula etc.
we will do a 5x cross validation.

Note, the time points (age at collection) are non standard, so we will have to bin them.
Note, this binning does not effect the model learning, only the visualisation after
(the model doesnt know about different time points)

So the plan is:
1. take a look at time points to see how we can bin them nicely.
2. do cross validation
3. visualise.


One of the big problems we will have to tackle is class imbalance...
I solved this here by turning this into a binary classification problem,
so its mothers milk vs other. I then also had to downsample mothers milk.
"""

#%%
import csv
import numpy as np

samples_path = 'data/samples.csv'
milk_path = 'data/milk.csv'

sample_rows = {}
with open(samples_path) as f:
    header = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        pieces = line.split(',')
        if header is None:
            header = pieces
            header[0] = header[0].lstrip('\ufeff')
            continue
        row = dict(zip(header, pieces))
        sample_rows[row['sampleID']] = row

milk_labels = {}
with open(milk_path) as f:
    header = None
    for line in f:
        line = line.strip()
        if not line:
            continue
        pieces = line.split(',')
        if header is None:
            header = pieces
            header[0] = header[0].lstrip('\ufeff')
            continue
        row = dict(zip(header, pieces))
        subject = row['subjectID']
        label = row['milk_first_three_days']
        if label:
            milk_labels[subject] = label

sample_records = []
ages = []
for srs, embedding in sample_embeddings.items():
    sample_info = micro_to_sample.get(srs, {})
    sample_id = sample_info.get('sample')
    subject_id = micro_to_subject.get(srs)
    if not sample_id or not subject_id:
        continue
    sample_row = sample_rows.get(sample_id)
    if not sample_row:
        continue
    age_str = sample_row.get('age_at_collection', '').strip()
    if not age_str:
        continue
    try:
        age_value = float(age_str)
    except ValueError:
        continue
    label_raw = milk_labels.get(subject_id)
    if not label_raw:
        continue
    label = 'mothers_breast_milk' if label_raw == 'mothers_breast_milk' else 'other'
    sample_records.append({
        'srs': srs,
        'subject': subject_id,
        'sample_id': sample_id,
        'age': age_value,
        'embedding': embedding.numpy(),
        'label': label
    })
    ages.append(age_value)

label_groups = {}
for record in sample_records:
    label_groups.setdefault(record['label'], []).append(record)

original_counts = {label: len(records) for label, records in label_groups.items()}
print('original class counts:', original_counts)

minority_count = min(original_counts.values())
rng = np.random.default_rng(config['cross_validation']['random_state'])
balanced_records = []
for label, records in label_groups.items():
    if len(records) > minority_count:
        indices = rng.choice(len(records), minority_count, replace=False)
        balanced_records.extend([records[i] for i in indices])
    else:
        balanced_records.extend(records)

sample_records = balanced_records

ages = np.array([record['age'] for record in sample_records], dtype=np.float32)
print('usable samples:', len(sample_records))
print('age range:', float(ages.min()), 'to', float(ages.max()))
print('age quantiles:', np.quantile(ages, [0, 0.25, 0.5, 0.75, 1.0]).tolist())
balanced_counts = {label: sum(1 for record in sample_records if record['label'] == label) for label in label_groups.keys()}
print('balanced class counts:', balanced_counts)

#%%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score

all_classes = sorted({record['label'] for record in sample_records})
print('milk categories:', all_classes)

bin_edges = np.quantile(ages, [0.0, 0.33, 0.66, 1.0])
bin_edges = np.unique(bin_edges)
if len(bin_edges) < 4:
    bin_edges = np.linspace(float(ages.min()), float(ages.max()), 4)
bin_ranges = []
for idx in range(len(bin_edges) - 1):
    start = float(bin_edges[idx])
    end = float(bin_edges[idx + 1])
    if idx < len(bin_edges) - 2:
        end -= 1e-6
    if start == end:
        continue
    bin_ranges.append((start, end))

bin_results = []
for start, end in bin_ranges:
    bin_records = [record for record in sample_records if start <= record['age'] <= end]
    if len(bin_records) < 10:
        print('Skipping bin', (start, end), 'because it has only', len(bin_records), 'samples.')
        continue

    X = np.stack([record['embedding'] for record in bin_records])
    y_labels = [record['label'] for record in bin_records]
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_labels)
    n_classes = len(label_encoder.classes_)

    if n_classes < 2:
        print('Skipping bin', (start, end), 'because it lacks class diversity.')
        continue

    skf = StratifiedKFold(
        n_splits=config['cross_validation']['k_folds'], 
        shuffle=True, 
        random_state=config['cross_validation']['random_state']
    )
    class_scores = {label: [] for label in label_encoder.classes_}
    macro_scores = []
    fold_predictions = []

    for train_idx, test_idx in skf.split(X, y):
        X_train = X[train_idx]
        X_test = X[test_idx]
        y_train = y[train_idx]
        y_test = y[test_idx]

        clf = LogisticRegression(
            max_iter=config['cross_validation']['max_iter'],
            solver=config['cross_validation']['solver'],
            class_weight=config['cross_validation']['class_weight']
        )
        clf.fit(X_train, y_train)
        probas = clf.predict_proba(X_test)
        y_pred = np.argmax(probas, axis=1)
        fold_predictions.append({
            'y_true': label_encoder.inverse_transform(y_test),
            'y_pred': label_encoder.inverse_transform(y_pred)
        })

        fold_auc_values = []
        for class_index, class_name in enumerate(label_encoder.classes_):
            binary_true = (y_test == class_index).astype(int)
            if binary_true.max() == binary_true.min():
                continue
            auc_value = roc_auc_score(binary_true, probas[:, class_index])
            class_scores[class_name].append(auc_value)
            fold_auc_values.append(auc_value)

        if n_classes == 2:
            if fold_auc_values:
                macro_scores.append(float(np.mean(fold_auc_values)))
        else:
            macro_scores.append(roc_auc_score(y_test, probas, multi_class='ovr', average='macro'))

    averaged_scores = {
        class_name: float(np.mean(scores)) if scores else None
        for class_name, scores in class_scores.items()
    }
    macro_mean = float(np.mean(macro_scores)) if macro_scores else None
    bin_results.append({
        'range': (start, end),
        'macro_auc': macro_mean,
        'per_class': averaged_scores,
        'macro_scores': macro_scores[:],
        'per_class_folds': {class_name: class_scores[class_name][:] for class_name in label_encoder.classes_},
        'fold_predictions': fold_predictions
    })
    print('bin', (start, end), 'macro AUC:', macro_mean, 'per class:', averaged_scores)

#%%
import matplotlib.pyplot as plt

bin_labels = [f'{round(r[0], 1)}-{round(r[1], 1)}' for r in [entry['range'] for entry in bin_results]]
class_names = sorted({name for entry in bin_results for name, score in entry['per_class'].items() if score is not None})
bin_positions = np.arange(len(bin_labels))

positive_label = 'mothers_breast_milk'

plt.figure(figsize=(8, 6))
macro_box_data = [entry['macro_scores'] for entry in bin_results]
plt.boxplot(macro_box_data, labels=bin_labels, showmeans=True)
plt.ylabel('ROC AUC')
plt.title('Macro ROC AUC per age bin (cross-validation folds)')
plt.xticks(rotation=45)
plt.axhline(0.5, color='gray', linestyle='--', label='random baseline')
plt.legend()
plt.tight_layout()
plt.show()


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, len(bin_results), figsize=(6 * len(bin_results), 5))
if len(bin_results) == 1:
    axes = [axes]

for ax, (entry, label) in zip(axes, zip(bin_results, bin_labels)):
    y_true_all = []
    y_pred_all = []
    for fold in entry['fold_predictions']:
        y_true_all.extend(fold['y_true'])
        y_pred_all.extend(fold['y_pred'])
    cm = confusion_matrix(y_true_all, y_pred_all, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion matrix for age bin {label}')
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')

plt.tight_layout()
plt.show()

# %%
"""
Okay thats pretty nice!
"""
