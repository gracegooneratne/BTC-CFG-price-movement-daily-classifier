"""
Bayesian Neural Network for Binary Classification (Up/Down price direction).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BayesianClassifier(nn.Module):
    """
    Bayesian Neural Network for binary classification of Bitcoin price direction.
    
    Output: 2 classes [Down=0, Up=1]
    """
    
    def __init__(self, 
                 input_size: int,
                 hidden_layers: list = [64, 32],
                 output_size: int = 2,
                 activation: str = 'relu',
                 dropout_rate: float = 0.3,
                 prior_scale: float = 1.0):
        """
        Args:
            input_size: Number of input features
            hidden_layers: List of hidden layer sizes
            output_size: Number of output classes (2 for binary)
            activation: Activation function ('relu', 'tanh', 'sigmoid')
            dropout_rate: Dropout probability
            prior_scale: Scale of prior distribution for Bayesian regularization
        """
        super(BayesianClassifier, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.prior_scale = prior_scale
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            
            layers.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size
        
        # Output layer (no activation - raw logits for CrossEntropyLoss)
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            logits: Raw scores [batch_size, output_size]
        """
        return self.network(x)
    
    def predict_proba(self, x):
        """
        Predict class probabilities.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            probs: Class probabilities [batch_size, output_size]
        """
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=1)
        return probs
    
    def predict(self, x):
        """
        Predict class labels.
        
        Args:
            x: Input tensor [batch_size, input_size]
            
        Returns:
            labels: Predicted class indices [batch_size]
        """
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)
    
    def get_num_parameters(self):
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def bayesian_loss(self, logits, targets, alpha=1.0, beta=0.001, class_weights=None):
        """
        Compute Bayesian loss with explicit non-linear L2 regularization.
        
        Per paper Eq. 4:
        E_B = (alpha/2) * Σ(t - o)^2 + (beta/2) * w_B^T * w_B
        
        Adapted for classification:
        E_B = alpha * CrossEntropy(t, o) + (beta/2) * w_B^T * w_B
        
        The regularization term (beta/2) * w^T * w is computed explicitly
        in the loss function (non-linear), NOT via optimizer weight_decay.
        This causes the regularization to interact with backpropagation
        non-linearly through the gradient computation.
        
        Args:
            logits: Model output [batch_size, output_size]
            targets: True labels [batch_size]
            alpha: Hyperparameter weighting data fit term
            beta: Hyperparameter weighting regularization term
            
        Returns:
            loss: Total Bayesian loss value
        """
        # Data fit term: cross-entropy loss (with optional class weights)
        ce_loss = F.cross_entropy(logits, targets, weight=class_weights)
        
        # Bayesian regularization: (beta/2) * w_B^T * w_B
        # Sum over ALL weight parameters (not biases, following convention)
        w_sq_sum = 0.0
        for name, param in self.named_parameters():
            if 'weight' in name:
                w_sq_sum += torch.sum(param ** 2)
        
        # Total Bayesian loss
        total_loss = alpha * ce_loss + (beta / 2.0) * w_sq_sum
        
        return total_loss
