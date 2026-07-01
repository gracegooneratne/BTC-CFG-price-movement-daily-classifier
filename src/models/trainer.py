"""
Trainer for the Bayesian Neural Network classifier.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Optional


class ClassifierTrainer:
    """Train and evaluate the BNN classifier."""
    
    def __init__(self, model, device: str = 'cpu', 
                 learning_rate: float = 0.001):
        """
        Args:
            model: BayesianClassifier model instance
            device: 'cpu' or 'cuda'
            learning_rate: Learning rate for optimizer
        """
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        
        # NO weight_decay in optimizer — regularization is computed
        # explicitly in bayesian_loss() per paper Eq. 4
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.0
        )
        self.history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    def fit(self, X_train, y_train, 
            batch_size: int = 32, epochs: int = 100,
            validation_split: float = 0.1,
            early_stopping_patience: int = 20,
            alpha: float = 1.0, beta: float = 0.01,
            use_class_weights: bool = False,
            verbose: bool = True):
        """
        Train the classifier.
        
        Args:
            X_train: Training features [n_samples, n_features]
            y_train: Training labels [n_samples] (0 or 1)
            batch_size: Batch size
            epochs: Number of training epochs
            validation_split: Fraction for validation
            early_stopping_patience: Epochs to wait before stopping
            alpha: Weight for cross-entropy term
            beta: Weight for L2 regularization term
            verbose: Print training progress
        """
        # Convert to tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train.ravel()).to(self.device)
        
        # Split into train/validation
        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size
        
        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]
        
        # Compute class weights if requested
        self._class_weights = None
        if use_class_weights:
            classes, counts = torch.unique(y_train, return_counts=True)
            weights = len(y_train) / (len(classes) * counts.float())
            self._class_weights = weights.to(self.device)
        
        # Create data loader
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        
        # Early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None
        
        self.model.train()
        
        for epoch in range(epochs):
            # Training
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.model.bayesian_loss(logits, y_batch, alpha=alpha, beta=beta, class_weights=self._class_weights)
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(X_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)
            
            train_loss = epoch_loss / total
            train_acc = correct / total
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = self.model.bayesian_loss(val_logits, y_val, alpha=alpha, beta=beta, class_weights=self._class_weights).item()
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == y_val).sum().item() / len(y_val)
            self.model.train()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            predictions: Class labels [n_samples]
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            preds = self.model.predict(X_tensor)
        return preds.cpu().numpy()
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Args:
            X: Features [n_samples, n_features]
            
        Returns:
            probabilities: Class probabilities [n_samples, 2]
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            probs = self.model.predict_proba(X_tensor)
        return probs.cpu().numpy()
