# big_brain/models/classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, MeanAbsoluteError, MeanSquaredError, ConfusionMatrix
import matplotlib.pyplot as plt
import seaborn as sns

from big_brain.models.transformer import DWIBert

class DWIBertFinetuner(pl.LightningModule):
    def __init__(
            self,
            # Model parameters
            bert_ckpt: str = "checkpoints/transformer_382_v1",
            task: str = "bin_cdr",  # "binary", "multiclass" or "regression"
            num_logits: int = 1,   # Number of classes for classification or 1 for regression
            # Training parameters
            freeze_depth: str | int = "all",  # "all" or an integer for the number of layers to freeze
            lr_classifier: float = 1e-3,
            lr_transformer: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load the pre-trained transformer model
        self.transformer = DWIBert.load_from_checkpoint(bert_ckpt)
        self.transformer._frozen = True  # Freeze the transformer layers by default

        # Freeze the specified depth of the transformer
        if freeze_depth == "all":
            for p in self.transformer.parameters():
                p.requires_grad = False
        elif isinstance(freeze_depth, int):
            for p in self.transformer.encoder.parameters():
                p.requires_grad = False
            # unfreeze the last k transformer layers
            for layer in self.transformer.encoder.layers[-freeze_depth:]:
                for p in layer.parameters():
                    p.requires_grad = True
        
        # Classifier head
        self.classifier = nn.Linear(self.transformer.hparams.d_model, num_logits)

        # Metrics
        self.task = task
        # Define the loss function and metrics for binary, multiclass, or regression tasks
        if task in ["bin_cdr", "gender"]:   # binary classification
            self.loss_fn = nn.BCEWithLogitsLoss()   # expects logits, so we don't apply sigmoid here (handled within the loss function)
            self.metrics = nn.ModuleDict({
                "auroc"    : AUROC(task="binary"),
                "accuracy" : Accuracy(task="binary"),
                "f1"       : F1Score(task="binary"),
                "precision": Precision(task="binary"),
                "recall"   : Recall(task="binary"),
            })
            self.cm_metric = ConfusionMatrix(task="binary", num_classes=2)
            if task == "bin_cdr":
                self.class_names = ["healthy", "impaired"]
            else:   # gender
                self.class_names = ["male", "female"]

        elif task in ["handedness", "tri_cdr"]: # multiclass classification
            self.loss_fn = nn.CrossEntropyLoss()    # expects logits, so we don't apply softmax here (handled within the loss function)
            self.metrics = nn.ModuleDict({
                "accuracy" : Accuracy(task="multiclass", num_classes=3),
                "f1"       : F1Score(task="multiclass",   num_classes=3, average="macro"),
                "precision": Precision(task="multiclass", num_classes=3, average="macro"),
                "recall"   : Recall(task="multiclass",    num_classes=3, average="macro"),
                "auroc"    : AUROC(task="multiclass",     num_classes=3, average="macro"),
            })
            self.cm_metric = ConfusionMatrix(task="multiclass", num_classes=3)
            if task == "handedness":
                self.class_names = ["right", "left", "both"]
            else:  # tri_cdr
                self.class_names = ["healthy", "MCI", "dementia"]

        else:  # regression
            self.loss_fn = nn.MSELoss()
            self.metrics = nn.ModuleDict({
                "mae" : MeanAbsoluteError(),
                "rmse": MeanSquaredError(squared=False),
            })
            self.cm_metric   = None  # No confusion matrix for regression
            self.class_names = None


    def forward(self, z, g, attn_mask):
        """
        Forward pass through the DWI-BERT model.
        Args:
            z (torch.Tensor): Latent vector of shape [B, L, 512].
            g (torch.Tensor): Gradient vector of shape [B, L, 4].
            attn_mask (torch.Tensor, optional): Attention mask of shape [B, L]. Defaults to None.
        Returns:
            
        """
        # project z and g to the embedding space
        x = self.transformer.embedder(z, g)             # Shape [B, L, D]
        x[:, 0] = x[:, 0] + self.transformer.cls_token  # Add [CLS] token at the start

        # Pass through the transformer encoder
        h = self.transformer.encoder(x, src_key_padding_mask=~attn_mask)

        # Compute the logits for classification
        logits = self.classifier(h[:, 0, :])  # Use the [CLS] token for classification

        return logits   # Shape [B, num_classes]


    def _step(self, batch, stage: str):
        """
        Common step for training, validation, and testing.
        Args:
            batch (tuple): A tuple containing (z, g, attn_mask, y).
            stage (str): The stage of the training process ('train', 'val', 'test').
        Returns:
            dict: A dictionary containing the loss and metrics.
        """
        # Unpack the batch
        z, g, attn_mask, y = batch

        # Forward pass
        logits = self.forward(z, g, attn_mask).squeeze(-1)

        # Compute loss for two cases (classification or regression):
        if self.task in ["bin_cdr", "gender"]:
            loss = self.loss_fn(logits.squeeze(), y.float())
            y_pred = torch.sigmoid(logits)              # probabilities
        elif self.task in ["handedness", "tri_cdr"]:
            loss = self.loss_fn(logits, y.long())
            y_pred = torch.softmax(logits, dim=1)       # probs per class
        else:  # regression
            logits = logits.squeeze()
            loss = self.loss_fn(logits, y.float())
            y_pred = logits                             # predicted ages (no activation)

        # Log the loss
        self.log(f"{stage}_loss", loss, prog_bar=True)

        return loss, y_pred, y
    

    def training_step(self, batch, batch_idx):
        """
        Training step.
        Args:
            batch (tuple): A tuple containing (z, g, attn_mask, y).
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value for the step.
        """
        loss, _, _ = self._step(batch, 'train')

        return loss
    

    def validation_step(self, batch, batch_idx):
        """
        Validation step.
        Args:
            batch (tuple): A tuple containing (z, g, attn_mask, y).
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value for the step.
        """
        # Perform the step and compute loss and predictions
        loss, y_pred, y = self._step(batch, 'val')

        # Update the scalar metrics
        for name, metric in self.metrics.items():
            metric(y_pred, y)
            # Only log epoch-level at validation
            self.log(f"val_{name}", metric, prog_bar=True, on_step=False, on_epoch=True)

        # Update the confusion matrix metric
        if self.cm_metric is not None:
            self.cm_metric.update(y_pred, y.long())

        return loss
    

    def test_step(self, batch, batch_idx):
        """
        Test step.
        Args:
            batch (tuple): A tuple containing (z, g, attn_mask, y).
            batch_idx (int): The index of the batch.
        Returns:
            torch.Tensor: The loss value for the step.
        """
        # Perform the step and compute loss and predictions
        loss, y_pred, y = self._step(batch, 'test')

        # Update the scaler metrics
        for name, metric in self.metrics.items():
            metric(y_pred, y)
            # Only log epoch-level at validation
            self.log(f"test_{name}", metric, prog_bar=True, on_step=False, on_epoch=True)

        # Update the confusion matrix metric
        if self.cm_metric is not None:
            self.cm_metric.update(y_pred, y.long())

        return loss
    
    def on_epoch_end(self):
        """
        log the confusion matrix at the end of the validation epoch.
        """
        if self.cm_metric is None:
            return
        
        # Plot the confusion matrix
        fig, ax = self.cm_metric.plot(labels=self.class_names, add_text=True, cmap="Blues")
        self.logger.experiment.add_figure("val_confusion_matrix", fig, self.current_epoch)

        # clean up
        plt.close(fig)
        self.cm_metric.reset()


    def on_test_epoch_end(self):
        return self.on_epoch_end()


    def on_validation_epoch_end(self):
        return self.on_epoch_end()
    

    def configure_optimizers(self):
        """
        Configure the optimizer and the learning rate scheduler for the model.
        Returns:
            torch.optim.Optimizer: The optimizer for the model.
        """
        # Use AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr_classifier,
            weight_decay=1e-2
        )

        return optimizer