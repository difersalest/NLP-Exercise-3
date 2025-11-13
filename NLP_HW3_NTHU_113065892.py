# %%
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import Counter
from transformers import BertTokenizer, BertModel
from datasets import load_dataset
from evaluate import load
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from tqdm import tqdm
device = "cuda" if torch.cuda.is_available() else "cpu"
#  You can install and import any other libraries if needed

# %%
# Some Chinese punctuations will be tokenized as [UNK], so we replace them with English ones
token_replacement = [
    ["：", ":"],
    ["，", ","],
    ["“", "\""],
    ["”", "\""],
    ["？", "?"],
    ["……", "..."],
    ["！", "!"]
]

# %%

tokenizer = BertTokenizer.from_pretrained(
    "google-bert/bert-base-uncased", cache_dir="./cache/")

# %%
tokenizer(text="Testing the output.", text_pair="Testing second sentence.", padding=True,
          truncation=True, return_tensors="pt", return_attention_mask=True, return_token_type_ids=True)

# %%


class SemevalDataset(Dataset):
    def __init__(self, split="train") -> None:
        super().__init__()
        assert split in ["train", "validation", "test"]
        self.data = load_dataset(
            "sem_eval_2014_task_1", split=split, trust_remote_code=True, cache_dir="./cache/"
        ).to_list()

    def __getitem__(self, index):
        d = self.data[index]
        # Replace Chinese punctuations with English ones
        for k in ["premise", "hypothesis"]:
            for tok in token_replacement:
                d[k] = d[k].replace(tok[0], tok[1])
        return d

    def __len__(self):
        return len(self.data)


data_sample = SemevalDataset(split="train").data[:3]
print(
    f"Dataset example: \n{data_sample[0]} \n{data_sample[1]} \n{data_sample[2]}")

# %%
train_data = SemevalDataset(split="train").data
train_data_df = pd.DataFrame(train_data)

# Counting the groundtruth labels in the data
class_counts = Counter(train_data_df["entailment_judgment"])
print(class_counts)

# Total training samples
num_samples = sum(class_counts.values())

# Calculating weights because the data is not balanced
class_weights = []
for i in range(len(class_counts)):
    # Inverse frequency weights
    weight = num_samples / (len(class_counts) * class_counts[i])
    class_weights.append(weight)

# Convert to a tensor to use when calculating the loss
class_weights_tensor = torch.tensor(
    class_weights, dtype=torch.float).to(device)

print(f"Calculated Class Weights: {class_weights_tensor}")

# %%
# Define the hyperparameters
# You can modify these values if needed
lr = 1e-5
epochs = 50
train_batch_size = 4
validation_batch_size = 4

# %%
# TODO1: Create batched data for DataLoader
# `collate_fn` is a function that defines how the data batch should be packed.
# This function will be called in the DataLoader to pack the data batch.


def collate_fn(batch):
    # TODO1-1: Implement the collate_fn function
    # Write your code here
    # The input parameter is a data batch (tuple), and this function packs it into tensors.
    # Use tokenizer to pack tokenize and pack the data and its corresponding labels.
    # Return the data batch and labels for each sub-task.
    premises = [data_instance["premise"] for data_instance in batch]
    hypothesis = [data_instance["hypothesis"] for data_instance in batch]
    relatedness_scores = [data_instance["relatedness_score"]
                          for data_instance in batch]
    entailment_judgments = [data_instance["entailment_judgment"]
                            for data_instance in batch]

    input_texts = tokenizer(text=premises, text_pair=hypothesis, padding=True, truncation=True,
                            return_tensors="pt", return_attention_mask=True, return_token_type_ids=True)

    relatedness_scores = torch.FloatTensor(relatedness_scores)
    entailment_judgments = torch.LongTensor(entailment_judgments)

    return input_texts, relatedness_scores, entailment_judgments


# TODO1-2: Define your DataLoader
dl_train = torch.utils.data.DataLoader(dataset=SemevalDataset(
    split="train"), collate_fn=collate_fn, batch_size=train_batch_size, shuffle=True, num_workers=32)  # Write your code here
dl_validation = torch.utils.data.DataLoader(dataset=SemevalDataset(
    split="validation"), collate_fn=collate_fn, batch_size=validation_batch_size, shuffle=False, num_workers=32)  # Write your code here
dl_test = torch.utils.data.DataLoader(dataset=SemevalDataset(split="test"), collate_fn=collate_fn,
                                      batch_size=validation_batch_size, shuffle=False, num_workers=32)  # Write your code here

# %%
print(next(iter(dl_train)))

# %%
# TODO2: Construct your model


class MultiLabelModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Write your code here
        # Define what modules you will use in the model
        # Please use "google-bert/bert-base-uncased" model (https://huggingface.co/google-bert/bert-base-uncased)
        # Besides the base model, you may design additional architectures by incorporating linear layers, activation functions, or other neural components.
        # Remark: The use of any additional pretrained language models is not permitted.

        num_classes = 3  # Based on our data
        self.bert_model = BertModel.from_pretrained(
            "google-bert/bert-base-uncased", cache_dir="./cache/")
        self.dropout = torch.nn.Dropout(0.3)

        hidden_size = 512
        self.intermediate_linear = torch.nn.Linear(
            self.bert_model.config.hidden_size, hidden_size)
        self.relu = torch.nn.ReLU()

        self.regression_head = torch.nn.Linear(hidden_size, 1)
        self.classification_head = torch.nn.Linear(hidden_size, num_classes)

    def forward(self, **kwargs):
        # Write your code here
        # Forward pass

        output = self.bert_model(input_ids=kwargs.get("input_ids"),
                                 token_type_ids=kwargs.get("token_type_ids"),
                                 attention_mask=kwargs.get("attention_mask"))
        output_dropout = self.dropout(output.pooler_output)
        intermediate_output = self.intermediate_linear(output_dropout)
        activated_output = self.relu(intermediate_output)

        output_regression = self.regression_head(activated_output)
        output_classification = self.classification_head(activated_output)

        return output_regression, output_classification

# %%
# TODO3: Define your optimizer and loss function


model = MultiLabelModel().to(device)
# TODO3-1: Define your Optimizer
optimizer = torch.optim.AdamW(
    params=model.parameters(), lr=lr)  # Write your code here

# TODO3-2: Define your loss functions (you should have two)
# Write your code here
classification_criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights_tensor)
regression_criterion = torch.nn.MSELoss()

# scoring functions
psr = load("pearsonr")
acc = load("accuracy")

# %%
best_score = 0.0

batches_loss = pd.DataFrame(columns=["epoch", "batch_no", "agg_loss"])
epoch_val_acc_corr = pd.DataFrame(
    columns=["epoch", "pearson_correlation", "accuracy"])

for ep in range(epochs):
    batch_train_index = 0
    pbar = tqdm(dl_train)
    pbar.set_description(f"Training epoch [{ep+1}/{epochs}]")
    model.train()
    # TODO4: Write the training loop
    # Write your code here
    # train your model
    # clear gradient
    # forward pass
    # compute loss
    # back-propagation
    # model optimization
    for input_batch, rel_score_batch, entail_judge_batch in pbar:
        input_batch = input_batch.to(device)
        rel_score_batch = rel_score_batch.to(device)
        entail_judge_batch = entail_judge_batch.to(device)

        optimizer.zero_grad()

        rel_score_preds, entail_judge_preds = model(**input_batch)

        regression_loss = regression_criterion(
            rel_score_preds.squeeze(), rel_score_batch)
        classification_loss = classification_criterion(
            entail_judge_preds, entail_judge_batch)

        overall_loss = regression_loss + classification_loss

        overall_loss.backward()
        optimizer.step()

        batch_train_index += 1
        if batch_train_index % 50 == 0:
            pbar.set_postfix(loss=overall_loss.item())

        if batch_train_index == 1:
            batches_loss = pd.concat([batches_loss, pd.DataFrame([[ep, batch_train_index, overall_loss.item()]], columns=[
                                     "epoch", "batch_no", "agg_loss"])], ignore_index=True)

        # Prints loss every 1000 batches
        if batch_train_index % 10 == 0:
            batches_loss = pd.concat([batches_loss, pd.DataFrame([[ep, batch_train_index, overall_loss.item()]], columns=[
                                     "epoch", "batch_no", "agg_loss"])], ignore_index=True)

    batches_loss = pd.concat([batches_loss, pd.DataFrame([[ep, batch_train_index, overall_loss.item()]], columns=[
                             "epoch", "batch_no", "agg_loss"])], ignore_index=True)
    pbar = tqdm(dl_validation)
    pbar.set_description(f"Validation epoch [{ep+1}/{epochs}]")
    model.eval()
    # TODO5: Write the evaluation loop
    # Write your code here
    # Evaluate your model
    # Output all the evaluation scores (PearsonCorr, Accuracy)
    real_rel_scores = []
    pred_rel_scores = []
    real_entail_classes = []
    pred_entail_classes = []

    with torch.no_grad():
        batch_val_index = 0
        for input_batch, rel_score_real_batch, entail_judge_real_batch in pbar:
            input_batch = input_batch.to(device)
            rel_score_real_batch = rel_score_real_batch.to(device)
            entail_judge_real_batch = entail_judge_real_batch.to(device)

            rel_score_pred_batch, entail_judge_pred_batch = model(
                **input_batch)

            entailment_predicted_labels = torch.argmax(
                entail_judge_pred_batch, dim=1)

            pred_rel_scores.append(rel_score_pred_batch.cpu())
            real_rel_scores.append(rel_score_real_batch.cpu())
            pred_entail_classes.append(entailment_predicted_labels.cpu())
            real_entail_classes.append(entail_judge_real_batch.cpu())

        pred_rel_scores = torch.cat(pred_rel_scores).squeeze()
        real_rel_scores = torch.cat(real_rel_scores)
        pred_entail_classes = torch.cat(pred_entail_classes)
        real_entail_classes = torch.cat(real_entail_classes)

        pearson_corr = psr.compute(references=real_rel_scores, predictions=pred_rel_scores)[
            'pearsonr']  # Write your code here
        accuracy = acc.compute(references=real_entail_classes, predictions=pred_entail_classes)[
            'accuracy']  # Write your code here
        # print(f"F1 Score: {f1.compute()}")
        # batch_val_index+=1
        # if batch_val_index%10==0:
        epoch_val_acc_corr = pd.concat([epoch_val_acc_corr, pd.DataFrame([[ep, pearson_corr, accuracy]], columns=[
                                       "epoch", "pearson_correlation", "accuracy"])], ignore_index=True)
        print(
            f"Epoch no. {ep} - Pearson Correlation: {pearson_corr} - Accuracy: {accuracy}")
        if pearson_corr + accuracy > best_score:
            best_score = pearson_corr + accuracy
            torch.save(model.state_dict(
            ), f'./saved_models/bert_multioutput_weighted_4_50_epochs_best_model.ckpt')

# %%
# Load the model
model = MultiLabelModel().to(device)
model.load_state_dict(torch.load(
    f"./saved_models/bert_multioutput_weighted_4_50_epochs_best_model.ckpt", weights_only=True))

# Test Loop
pbar = tqdm(dl_test, desc="Test")
model.eval()

# TODO6: Write the test loop
# Write your code here
# We have loaded the best model with the highest evaluation score for you
# Please implement the test loop to evaluate the model on the test dataset
# We will have 10% of the total score for the test accuracy and pearson correlation
real_rel_scores = []
pred_rel_scores = []
real_entail_classes = []
pred_entail_classes = []

with torch.no_grad():
    batch_val_index = 0
    for input_batch, rel_score_real_batch, entail_judge_real_batch in pbar:
        input_batch = input_batch.to(device)
        rel_score_real_batch = rel_score_real_batch.to(device)
        entail_judge_real_batch = entail_judge_real_batch.to(device)

        rel_score_pred_batch, entail_judge_pred_batch = model(**input_batch)

        entailment_predicted_labels = torch.argmax(
            entail_judge_pred_batch, dim=1)

        pred_rel_scores.append(rel_score_pred_batch.cpu())
        real_rel_scores.append(rel_score_real_batch.cpu())
        pred_entail_classes.append(entailment_predicted_labels.cpu())
        real_entail_classes.append(entail_judge_real_batch.cpu())

    pred_rel_scores = torch.cat(pred_rel_scores).squeeze()
    real_rel_scores = torch.cat(real_rel_scores)
    pred_entail_classes = torch.cat(pred_entail_classes)
    real_entail_classes = torch.cat(real_entail_classes)

    pearson_corr = psr.compute(
        references=real_rel_scores, predictions=pred_rel_scores)['pearsonr']
    accuracy = acc.compute(references=real_entail_classes,
                           predictions=pred_entail_classes)['accuracy']

    test_results_df = pd.DataFrame([["Test Set", pearson_corr, accuracy]], columns=[
                                   "set", "pearson_correlation", "accuracy"])
    print(
        f"Test Set - Pearson Correlation: {pearson_corr} - Accuracy: {accuracy}")

# %% [markdown]
# ### Processing results for graphs:

# %%
melted_epoch_test_acc_corr = test_results_df.melt(
    id_vars="set", var_name="Type", value_name="Value")
melted_epoch_test_acc_corr

# %%
melted_epoch_val_acc_corr = epoch_val_acc_corr.melt(
    id_vars="epoch", var_name="Type", value_name="Value")
melted_epoch_val_acc_corr

# %%
sns.set_theme()

output_dir = "./results/bert_multioutput_weighted_4_50_epochs_model_results"
os.makedirs(output_dir, exist_ok=True)

batches_loss['global_batch'] = range(len(batches_loss))

# Calculate the position for epoch markers
epoch_boundaries = batches_loss.groupby('epoch')['global_batch'].min().tolist()
epoch_boundaries.append(len(batches_loss))

# Get the center of each epoch
epoch_centers = [(epoch_boundaries[i] + epoch_boundaries[i+1]
                  ) / 2 for i in range(len(epoch_boundaries)-1)]


# Create the plot
fig, ax = plt.subplots(3, 1, figsize=(15, 20))

ax1 = ax[0]
# Plot batch loss on the primary y-axis
color = 'tab:red'
ax1.set_xlabel('Global Batch Number')
ax1.set_ylabel('Batch Aggregated Loss', color=color)
sns.lineplot(data=batches_loss, x='global_batch', y='agg_loss',
             ax=ax1, color=color, alpha=0.7, label='Batch Aggregated Loss')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim(bottom=min(batches_loss["agg_loss"]))


ax2 = ax[1]
# Plot batch loss on the primary y-axis
color = 'black'
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Validation Correlation/Accuracy', color=color)
sns.barplot(data=melted_epoch_val_acc_corr,
            x="epoch", y="Value", hue="Type", ax=ax2)
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_ylim(min([min(epoch_val_acc_corr["pearson_correlation"]),
             min(epoch_val_acc_corr["accuracy"])])-0.05, 1)

ax3 = ax[2]

color = "black"
ax3.set_ylabel('Test Correlation/Accuracy', color=color)
sns.barplot(data=melted_epoch_test_acc_corr,
            x="set", y="Value", hue="Type", ax=ax3)
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_ylim(min([min(test_results_df["pearson_correlation"]),
             min(test_results_df["accuracy"])])-0.05, 1)
ax3.text("Test Set", ax3.get_ylim()[
         1] * 0.90, f'Test Set\nAcc={test_results_df.iloc[0]["accuracy"]:.3f}\nCorr={test_results_df.iloc[0]["pearson_correlation"]:.3f}', horizontalalignment='center', color='black')

# Add vertical lines and text for epoch boundaries
for i, (boundary, epoch) in enumerate(zip(epoch_boundaries[:-1], epoch_val_acc_corr["epoch"])):
    # for i, boundary in enumerate(epoch_boundaries[:-1]):
    epoch_align = epoch
    epoch_sep = epoch+0.5
    ax1.axvline(x=boundary, color='gray', linestyle='--', linewidth=1)
    ax2.axvline(x=epoch_sep, color='gray', linestyle='--', linewidth=1)
    # Add text label for the epoch
    # Too many epochs, so this line was commented out to not clutter the plot
    # ax2.text(epoch_align, ax2.get_ylim()[1] * 0.90, f'Epoch {i+1}\nAcc={epoch_val_acc_corr.iloc[i]["accuracy"]:.3f}\nCorr={epoch_val_acc_corr.iloc[i]["pearson_correlation"]:.3f}', horizontalalignment='center', color='black')

ax1.set_title('BERT Multi-Output Weighted Model Results - Intermediate Linear Layer with Batch 4 and LR 1e-5 - 50 Epochs\n\nBatch Aggregated Loss')
ax2.set_title(
    'Validation Set \nRelatedness Scores Correlations & Entailment Judgement Accuracy')
ax3.set_title(
    'Test Set \nRelatedness Scores Correlations & Entailment Judgement Accuracy')
fig.tight_layout()
plt.savefig(f"{output_dir}/reg_class_loss_acc_corr_model.png")

plt.tight_layout()

# %%
# Saving the results data
batches_loss.to_csv(f"{output_dir}/batches_loss.csv")
epoch_val_acc_corr.to_csv(f"{output_dir}/epoch_val_acc_corr.csv")

# %% [markdown]
# ### Error analysis:

# %%
# Matching sentences pairs with results
test_data = SemevalDataset(split="test").data
test_data_df = pd.DataFrame(test_data)
results_test_df = pd.DataFrame({"Real_Rel_Scores": real_rel_scores, "Pred_Rel_Scores": pred_rel_scores,
                               "Real_Entail_Classes": real_entail_classes, "Pred_Entail_Classes": pred_entail_classes})
results_test_df = pd.concat([test_data_df, results_test_df], axis=1)

# Sorting by real relatedness scores values
sorted_results_test_df = results_test_df.sort_values(
    by="Real_Rel_Scores", ignore_index=True)
sorted_results_test_df["index"] = range(0, len(sorted_results_test_df))

# Making the map from number labels to the sentence classes
label_map = {0: "Neutral", 1: "Entailment", 2: "Contradiction"}
sorted_results_test_df["Real_Entail_Classes_Map"] = sorted_results_test_df["Real_Entail_Classes"].apply(
    lambda x: label_map[x])
sorted_results_test_df["Pred_Entail_Classes_Map"] = sorted_results_test_df["Pred_Entail_Classes"].apply(
    lambda x: label_map[x])
print(sorted_results_test_df.head(5))

# Obtaining the counts of real and predicted classes
count_labels_real = Counter(sorted_results_test_df["Real_Entail_Classes_Map"])
count_labels_pred = Counter(sorted_results_test_df["Pred_Entail_Classes_Map"])
count_labels_real_df = pd.DataFrame({"Label": count_labels_real.keys(
), "Groundtruth": count_labels_real.values()}, columns=["Label", "Groundtruth"])
count_labels_pred_df = pd.DataFrame({"Label": count_labels_pred.keys(
), "Predictions": count_labels_pred.values()}, columns=["Label", "Predictions"])
count_labels_df = count_labels_real_df.merge(count_labels_pred_df, on="Label")
print(count_labels_df)

# Preparing the data to plot
melted_df = count_labels_df.melt(
    id_vars="Label", var_name="Type", value_name="Count")
print(melted_df)


# %%
sorted_results_test_df.to_csv(f"{output_dir}/full_results_test_set.csv")

# %%
sns.set_theme()

# Plotting relatedness scores and judgement entailment real vs predicted
fig, ax = plt.subplots(2, 1, figsize=(10, 10))
fig.dpi = 100

sns.scatterplot(data=sorted_results_test_df, x="index",
                y="Pred_Rel_Scores", color="red", label="Predictions", s=4, ax=ax[0])
sns.lineplot(data=sorted_results_test_df, x="index",
             y="Real_Rel_Scores", color="blue", label="Groundtruth", ax=ax[0])

ax[0].set_ylim(0, max(sorted_results_test_df["Pred_Rel_Scores"])+2)
ax[0].set_xlabel("Index")
ax[0].set_ylabel("Relatedness Score")
ax[0].set_title("BERT Multi-Output Weighted Model Results - Intermediate Linear Layer with Batch 4 and LR 1e-5 - 50 Epochs \n\n Sorted Relatedness Scores - Groundtruth vs Predictions")

sns.barplot(data=melted_df, x="Label", y="Count",
            hue="Type", ax=ax[1], alpha=0.7)
ax[1].bar_label(ax[1].containers[0])
ax[1].bar_label(ax[1].containers[1])
ax[1].set_xlabel("Entailment Judgement")
ax[1].set_ylabel("Count")
ax[1].set_title("Entailment Judgement - Groundtruth vs Predictions")
fig.tight_layout()
plt.savefig(f"{output_dir}/rel_score_ent_judg_graphs.png")

# %%

# Plotting confusion matrix
cm = confusion_matrix(
    sorted_results_test_df["Real_Entail_Classes_Map"], sorted_results_test_df["Pred_Entail_Classes_Map"])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[
                              "Neutral", "Entailment", "Contradiction"])
disp.plot()
plt.title("Confusion Matrix - Entailment Judgement")
plt.grid(visible=False)
plt.savefig(f"{output_dir}/confusion_matrix_ent_judge.png")
plt.show()

# %%
# Getting classification report from the results
report = classification_report(
    sorted_results_test_df["Real_Entail_Classes_Map"], sorted_results_test_df["Pred_Entail_Classes_Map"])
print(report)

# %%
sorted_results_test_df.describe()  # Statistics of the data

# %%
# Sorting for top 5 highest difference between real and predicted relatedness scores
sorted_results_test_df["Rel_Score_Diff"] = sorted_results_test_df.apply(
    lambda row: abs(row.Real_Rel_Scores-row.Pred_Rel_Scores), axis=1)
sorted_rel_score_diff_df = sorted_results_test_df.sort_values(
    by="Rel_Score_Diff", ignore_index=True)
sorted_rel_score_diff_df[-5:]
