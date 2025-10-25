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
Okay, now, lets try and see if we can predict HLA.
We have a few different HLA risk classes. 
We will again have to bin by time, although the model wont know this.
We will again face class imbalace.
I will simplify by dropping the smallest class and balancing the rest with down sampling.
"""
#%% Collect sample-level records with HLA class and age.
import numpy as np

samples_path = 'data/samples.csv'
hla_path = 'data/pregnancy_birth.csv'

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

hla_labels = {}
with open(hla_path) as f:
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
        label = row['HLA_risk_class']
        if label:
            hla_labels[subject] = label

sample_records = []
ages = []
allowed_labels = {'2', '3'}
label_counts = {}
for srs, sample_info in micro_to_sample.items():
    subject_id = micro_to_subject.get(srs)
    if not subject_id:
        continue
    hla_label = hla_labels.get(subject_id)
    if hla_label not in allowed_labels:
        continue
    sample_id = sample_info.get('sample')
    if not sample_id:
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
    sample_records.append({
        'srs': srs,
        'subject': subject_id,
        'sample_id': sample_id,
        'age': age_value,
        'label': hla_label
    })
    ages.append(age_value)
    label_counts[hla_label] = label_counts.get(hla_label, 0) + 1

ages = np.array(ages, dtype=np.float32)
print('sample count with HLA labels:', len(sample_records))
print('age range:', float(ages.min()), 'to', float(ages.max()))
print('age quantiles:', np.quantile(ages, [0, 0.25, 0.5, 0.75, 1.0]).tolist())
print('class counts (samples):', label_counts)

# Keep only records that have model embeddings.
model_records = []
missing_embeddings = 0
for record in sample_records:
    if record['srs'] in sample_embeddings:
        model_records.append(record)
    else:
        missing_embeddings += 1

print('records with embeddings:', len(model_records))
print('records missing embeddings:', missing_embeddings)

# Downsample to match the minority class.
label_groups = {label: [] for label in allowed_labels}
for record in model_records:
    label_groups[record['label']].append(record)

minority_count = min(len(records) for records in label_groups.values() if records)
rng = np.random.default_rng(config['cross_validation']['random_state'])
balanced_records = []
for label, records in label_groups.items():
    if not records:
        continue
    if len(records) > minority_count:
        indices = rng.choice(len(records), minority_count, replace=False)
        balanced_records.extend(records[i] for i in indices)
    else:
        balanced_records.extend(records)

model_records = [balanced_records[i] for i in rng.permutation(len(balanced_records))]
balanced_counts = {label: sum(1 for record in model_records if record['label'] == label) for label in allowed_labels}
print('balanced class counts (with embeddings):', balanced_counts)

# Compute age bins identical to the previous workflow.
bin_edges = np.quantile(ages, [0.0, 0.33, 0.66, 1.0])
bin_edges = np.unique(bin_edges)
if len(bin_edges) < 4:
    bin_edges = np.linspace(float(ages.min()), float(ages.max()), 4)
age_bins = []
for idx in range(len(bin_edges) - 1):
    start = float(bin_edges[idx])
    end = float(bin_edges[idx + 1])
    if idx < len(bin_edges) - 2:
        end -= 1e-6
    if start == end:
        continue
    age_bins.append((start, end))

bin_labels = [f'{round(r[0], 1)}-{round(r[1], 1)}' for r in age_bins]
print('age bins:', bin_labels)

# Count how many samples fall into each HLA class per age bin.
bin_hla_counts = []
for start, end in age_bins:
    bin_records = [record for record in sample_records if start <= record['age'] <= end]
    counts = {}
    for record in bin_records:
        counts[record['label']] = counts.get(record['label'], 0) + 1
    bin_hla_counts.append({'range': (start, end), 'counts': counts})
    print('bin', (start, end), 'counts:', counts)

#%% Fit a balanced binary logistic regression (class 3 vs class 2).
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score

label_encoder = LabelEncoder()
label_encoder.fit(sorted(allowed_labels))

X = []
y_encoded = []
bin_index = []
for record in model_records:
    embedding = sample_embeddings[record['srs']].numpy()
    X.append(embedding)
    y_encoded.append(label_encoder.transform([record['label']])[0])
    age_value = record['age']
    for idx, (start, end) in enumerate(age_bins):
        if start <= age_value <= end:
            bin_index.append(idx)
            break

X = np.stack(X)
y_encoded = np.array(y_encoded, dtype=np.int64)
bin_index = np.array(bin_index, dtype=np.int64)

prob_scores = np.zeros(len(y_encoded), dtype=np.float32)
pred_labels = np.zeros(len(y_encoded), dtype=np.int64)

skf = StratifiedKFold(
    n_splits=config['cross_validation']['k_folds'], 
    shuffle=True, 
    random_state=config['cross_validation']['random_state']
)
for train_idx, test_idx in skf.split(X, y_encoded):
    clf = LogisticRegression(
        max_iter=config['cross_validation']['max_iter'],
        solver=config['cross_validation']['solver'],
        class_weight=config['cross_validation']['class_weight']
    )
    clf.fit(X[train_idx], y_encoded[train_idx])
    probas = clf.predict_proba(X[test_idx])[:, 1]
    prob_scores[test_idx] = probas
    pred_labels[test_idx] = (probas >= 0.5).astype(np.int64)

overall_acc = float(np.mean(pred_labels == y_encoded))
overall_auc = roc_auc_score(y_encoded, prob_scores)
print('overall accuracy:', overall_acc)
print('overall ROC AUC:', overall_auc)
print('classification report:')
print(classification_report(
    label_encoder.inverse_transform(y_encoded),
    label_encoder.inverse_transform(pred_labels),
    target_names=label_encoder.classes_,
    digits=3,
    zero_division=0
))

bin_accuracies = {}
for idx, label in enumerate(bin_labels):
    mask = bin_index == idx
    if not np.any(mask):
        bin_accuracies[label] = None
        continue
    bin_true = y_encoded[mask]
    bin_pred = pred_labels[mask]
    bin_acc = float(np.mean(bin_pred == bin_true))
    bin_accuracies[label] = bin_acc
    print('bin', label, 'accuracy:', bin_acc)

#%%
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(1, len(bin_labels), figsize=(6 * len(bin_labels), 5))
if len(bin_labels) == 1:
    axes = [axes]

for ax, label, idx in zip(axes, bin_labels, range(len(bin_labels))):
    mask = bin_index == idx
    if not np.any(mask):
        ax.axis('off')
        ax.set_title(f'Age bin {label}\n(no samples)')
        continue
    true_labels = label_encoder.inverse_transform(y_encoded[mask])
    pred_labels_display = label_encoder.inverse_transform(pred_labels[mask])
    cm = confusion_matrix(true_labels, pred_labels_display, labels=label_encoder.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Age bin {label}')
    ax.tick_params(axis='x', rotation=45)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

plt.tight_layout()
plt.show()

# %%

"""
Okay so we are a bit better than random! still, this is a hard task. We have no 
'baseline' of healthy and to be honest Im not sure how different risk class 2 and 3 are..
The model seems to do best at predicted in the oldest age group which is interesting.
"""
