"""
Interactive weight visualization for the BayesianClassifier.

Produces:
  - results/weights_final.png   — static heatmap of final-epoch weights
  - results/weights_animation.gif — animated GIF of weights evolving over epochs

Uses a subclass of ClassifierTrainer to capture weight snapshots at every epoch
without modifying source files.
"""

import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.patches import Polygon

plt.style.use('dark_background')

from src.data.data_loader import DailyBitcoinDataLoader
from src.data.preprocessor import DailyDataPreprocessor
from src.models.bayesian_classifier import BayesianClassifier
from src.models.trainer import ClassifierTrainer
from src.utils.config_loader import load_config, get_feature_names


class SnapshotTrainer(ClassifierTrainer):
    """
    Subclass of ClassifierTrainer that captures weight snapshots
    at every epoch during training.
    """

    def __init__(self, model, device='cpu', learning_rate=0.001):
        super().__init__(model, device=device, learning_rate=learning_rate)
        self.weight_history = []

    def _snapshot_weights(self, epoch, train_loss=None):
        """Capture a snapshot of all weight matrices and training loss."""
        layers = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                layers[name] = param.data.cpu().numpy().copy()
        self.weight_history.append({'epoch': epoch, 'layers': layers, 'train_loss': train_loss})

    def fit(self, X_train, y_train,
            batch_size=32, epochs=100,
            validation_split=0.1,
            early_stopping_patience=20,
            alpha=1.0, beta=0.01,
            use_class_weights=False,
            verbose=True):
        """
        Override fit() to inject weight snapshots after every epoch.
        Reimplements the training loop with snapshot hooks.
        """
        from torch.utils.data import DataLoader, TensorDataset

        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train.ravel()).to(self.device)

        val_size = int(len(X_train) * validation_split)
        train_size = len(X_train) - val_size

        X_val = X_train[train_size:]
        y_val = y_train[train_size:]
        X_train = X_train[:train_size]
        y_train = y_train[:train_size]

        self._class_weights = None
        if use_class_weights:
            classes, counts = torch.unique(y_train, return_counts=True)
            weights = len(y_train) / (len(classes) * counts.float())
            self._class_weights = weights.to(self.device)

        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        best_state = None

        # Snapshot initial weights (epoch 0, before training)
        self._snapshot_weights(0)

        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                logits = self.model(X_batch)
                loss = self.model.bayesian_loss(
                    logits, y_batch, alpha=alpha, beta=beta,
                    class_weights=self._class_weights
                )
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item() * len(X_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()
                total += len(y_batch)

            train_loss = epoch_loss / total
            train_acc = correct / total

            self.model.eval()
            with torch.no_grad():
                val_logits = self.model(X_val)
                val_loss = self.model.bayesian_loss(
                    val_logits, y_val, alpha=alpha, beta=beta,
                    class_weights=self._class_weights
                ).item()
                val_preds = torch.argmax(val_logits, dim=1)
                val_acc = (val_preds == y_val).sum().item() / len(y_val)
            self.model.train()

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)

            # Snapshot weights after this epoch
            self._snapshot_weights(epoch + 1, train_loss=train_loss)

            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f} Acc: {train_acc:.4f} - "
                      f"Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}")

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

        if best_state is not None:
            self.model.load_state_dict(best_state)

        print(f"Captured {len(self.weight_history)} weight snapshots "
              f"(epoch 0 through {self.weight_history[-1]['epoch']})")


def get_layer_info(weight_layers):
    """
    Parse weight layer names and return ordered list of
    (name, display_name, in_size, out_size).
    """
    info = []
    sorted_names = sorted(weight_layers.keys())
    for i, name in enumerate(sorted_names):
        w = weight_layers[name]
        out_size, in_size = w.shape
        if i == len(sorted_names) - 1:
            display = f"Layer {i+1} Weights\nin {in_size}, out {out_size} (classes)"
        else:
            display = f"Layer {i+1} Weights\nin {in_size}, out {out_size}"
        info.append((name, display, in_size, out_size))
    return info


def compute_global_vrange(weight_history):
    """Compute global min/max across ALL epochs and layers for consistent colormap."""
    vmin = float('inf')
    vmax = float('-inf')
    for snap in weight_history:
        for w in snap['layers'].values():
            vmin = min(vmin, w.min())
            vmax = max(vmax, w.max())
    mag = max(abs(vmin), abs(vmax))
    return -mag, mag


def compute_delta_vrange(weight_history):
    """Compute global min/max of weight DELTAS from epoch 0 for animation colormap."""
    init_layers = weight_history[0]['layers']
    vmin = 0.0
    vmax = 0.0
    for snap in weight_history[1:]:
        for name, w in snap['layers'].items():
            delta = w - init_layers[name]
            vmin = min(vmin, delta.min())
            vmax = max(vmax, delta.max())
    mag = max(abs(vmin), abs(vmax))
    if mag == 0:
        mag = 1e-6
    return -mag, mag


def plot_weight_frame(fig, axes, caxes, snapshot, init_layers, layer_info, vmin, vmax, epoch_label):
    """Plot a single frame of weight delta heatmaps (change from epoch 0)."""
    for idx, (ax, cax, (name, display, in_size, out_size)) in enumerate(
            zip(axes, caxes, layer_info)):
        ax.clear()
        cax.clear()

        delta = snapshot['layers'][name] - init_layers[name]
        # Transpose: input features on Y axis, neurons on X axis
        im = ax.imshow(delta.T, aspect='auto', cmap='RdYlGn',
                        vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_title(display, color='white', fontsize=10, pad=8)
        ax.set_xlabel('Output neurons', color='white', fontsize=8)
        ax.set_ylabel('Input features', color='white', fontsize=8)
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('white')

        fig.colorbar(im, cax=cax)
        cax.tick_params(colors='white', labelsize=7)
        for spine in cax.spines.values():
            spine.set_color('white')

    fig.suptitle(epoch_label, color='white', fontsize=14, y=0.98)


def create_static_plot(weight_history, output_path):
    """Create static heatmap of final epoch weights."""
    final_snap = weight_history[-1]
    layer_info = get_layer_info(final_snap['layers'])
    n_layers = len(layer_info)

    vmin, vmax = compute_global_vrange(weight_history)

    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 8))
    fig.patch.set_facecolor('black')
    if n_layers == 1:
        axes = [axes]

    for idx, (ax, (name, display, in_size, out_size)) in enumerate(
            zip(axes, layer_info)):
        ax.set_facecolor('black')
        w = final_snap['layers'][name]
        im = ax.imshow(w.T, aspect='auto', cmap='RdYlGn',
                        vmin=vmin, vmax=vmax, interpolation='nearest')

        ax.set_title(display, color='white', fontsize=12, pad=10)
        ax.set_xlabel('Output neurons', color='white', fontsize=9)
        ax.set_ylabel('Input features', color='white', fontsize=9)
        ax.tick_params(colors='white', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#444444')

        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='white', labelsize=7)
        cbar.outline.set_edgecolor('#444444')

    epoch_num = final_snap['epoch']
    fig.suptitle(f'BNN Weight Heatmaps \u2014 Final Epoch {epoch_num}',
                 color='white', fontsize=14, y=1.02)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight',
                facecolor='black', edgecolor='none')
    plt.close(fig)
    print(f"Static plot saved to {output_path}")


def create_animation(weight_history, output_path):
    """Create animated GIF of weight DELTAS from epoch 0, evolving over epochs."""
    first_snap = weight_history[0]
    init_layers = first_snap['layers']
    layer_info = get_layer_info(first_snap['layers'])
    n_layers = len(layer_info)

    # Use delta range so changes are visible
    vmin, vmax = compute_delta_vrange(weight_history)
    print(f"Animation delta range: [{vmin:.6f}, {vmax:.6f}]")

    # Compute loss range across all epochs (skip epoch 0 which has None)
    all_losses = [s['train_loss'] for s in weight_history if s['train_loss'] is not None]
    loss_min = min(all_losses) if all_losses else 0
    loss_max = max(all_losses) if all_losses else 1
    if loss_min == loss_max:
        loss_max = loss_min + 1
    print(f"Loss range: [{loss_min:.2f}, {loss_max:.2f}]")

    # Create figure — wider to accommodate loss bar on the right
    fig = plt.figure(figsize=(6 * n_layers + 2.5, 8))
    fig.patch.set_facecolor('black')

    # Total width units for layout
    total_w = n_layers * 2 + 0.5 + 1.2  # extra 1.2 for loss bar region

    # Create axes grid: one main ax + one colorbar ax per layer
    axes = []
    caxes = []
    for i in range(n_layers):
        left = (i * 2 + 0.3) / total_w
        width = 1.4 / total_w
        ax = fig.add_axes([left, 0.12, width, 0.75])
        axes.append(ax)
        cax = fig.add_axes([left + width + 0.005, 0.12, 0.012, 0.75])
        caxes.append(cax)

    # Loss bar axis — tall narrow bar on the far right
    loss_bar_left = (n_layers * 2 + 0.5) / total_w
    loss_bar_width = 0.3 / total_w
    loss_bar_ax = fig.add_axes([loss_bar_left, 0.12, loss_bar_width, 0.75])

    # Pointer/label area to the right of the bar
    pointer_left = loss_bar_left + loss_bar_width
    pointer_width = 0.8 / total_w
    pointer_ax = fig.add_axes([pointer_left, 0.12, pointer_width, 0.75])

    total_epochs = weight_history[-1]['epoch']

    # Pre-render the gradient image for the loss bar (red top, green bottom)
    gradient = np.linspace(1, 0, 256).reshape(256, 1)  # 1=red(top), 0=green(bottom)
    cmap_bar = plt.cm.RdYlGn  # green=low values, red=high values; we want red top

    def update(frame_idx):
        snap = weight_history[frame_idx]
        epoch_label = f'Weight Changes from Init \u2014 Epoch {snap["epoch"]}/{total_epochs}'
        plot_weight_frame(fig, axes, caxes, snap, init_layers, layer_info,
                          vmin, vmax, epoch_label)

        # --- Draw loss bar ---
        loss_bar_ax.clear()
        pointer_ax.clear()

        # Draw gradient bar (green bottom to red top)
        loss_bar_ax.imshow(gradient, aspect='auto', cmap='RdYlGn_r',
                           extent=[0, 1, 0, 1], interpolation='bilinear')
        loss_bar_ax.set_xlim(0, 1)
        loss_bar_ax.set_ylim(0, 1)
        loss_bar_ax.set_xticks([])
        loss_bar_ax.set_yticks([])
        for spine in loss_bar_ax.spines.values():
            spine.set_color('white')
            spine.set_linewidth(1.5)

        # Labels
        loss_bar_ax.text(0.5, 1.03, 'High\nloss', ha='center', va='bottom',
                         color='white', fontsize=8, transform=loss_bar_ax.transAxes)
        loss_bar_ax.text(0.5, -0.03, 'Low\nloss', ha='center', va='top',
                         color='white', fontsize=8, transform=loss_bar_ax.transAxes)

        # Pointer area styling
        pointer_ax.set_facecolor('black')
        pointer_ax.set_xlim(0, 1)
        pointer_ax.set_ylim(0, 1)
        pointer_ax.set_xticks([])
        pointer_ax.set_yticks([])
        pointer_ax.axis('off')

        # Draw pointer if we have a loss value
        current_loss = snap['train_loss']
        if current_loss is not None:
            # Normalise: 0=bottom(low), 1=top(high)
            norm_loss = (current_loss - loss_min) / (loss_max - loss_min)
            norm_loss = np.clip(norm_loss, 0, 1)
            y_pos = 1.0 - norm_loss  # invert: high loss = top of bar = low y in axes
            # Actually: y=0 is bottom, y=1 is top. High loss should be near top.
            y_pos = norm_loss  # high loss = high y = top

            # Draw left-pointing triangle in pointer_ax coordinates
            tri_h = 0.03  # half-height of triangle
            tri = Polygon(
                [[0.0, y_pos],           # left tip (pointing left)
                 [0.35, y_pos + tri_h],   # top-right
                 [0.35, y_pos - tri_h]],  # bottom-right
                closed=True, facecolor='white', edgecolor='white',
                transform=pointer_ax.transAxes, clip_on=False
            )
            pointer_ax.add_patch(tri)

            # Loss value label next to pointer (compact formatting)
            if abs(current_loss) >= 1e6:
                loss_str = f'{current_loss:.2e}'
            else:
                loss_str = f'{current_loss:.2f}'
            pointer_ax.text(0.45, y_pos, loss_str,
                            color='white', fontsize=8, va='center', ha='left',
                            transform=pointer_ax.transAxes)

        return []

    anim = animation.FuncAnimation(
        fig, update, frames=len(weight_history),
        interval=100, blit=False  # 100ms = 10fps
    )

    print(f"Rendering animation with {len(weight_history)} frames...")
    anim.save(output_path, writer='pillow', fps=10,
              savefig_kwargs={'facecolor': 'black', 'edgecolor': 'none'})
    plt.close(fig)
    print(f"Animation saved to {output_path}")


def main():
    print("="*60)
    print("BNN WEIGHT VISUALIZATION")
    print("="*60)

    # Load config and best Optuna params
    config = load_config('config/config.yaml')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    with open('results/optuna_bc_results_v2.json', 'r') as f:
        optuna_results = json.load(f)

    best_params = optuna_results['best_params']
    hidden_layers = optuna_results['hidden_layers']

    print(f"\nUsing Optuna best hyperparameters (Trial #{optuna_results['best_trial_number']}):")
    print(f"  Hidden layers: {hidden_layers}")
    print(f"  Activation: {best_params['activation']}")
    print(f"  Dropout: {best_params['dropout_rate']:.4f}")
    print(f"  Epochs: {best_params['epochs']}")

    # Load and prepare data
    loader = DailyBitcoinDataLoader(config['data']['raw_dir'])
    df = loader.load_and_prepare_data(filename=config['data']['daily_csv'],
                                       add_lags=True, lag_days=1)

    feature_names = get_feature_names(config)
    for col in df.columns:
        if col.startswith('Close_lag') or col.startswith('PriceChange_lag'):
            if col not in feature_names:
                feature_names.append(col)
    available = [f for f in feature_names if f in df.columns]

    preprocessor = DailyDataPreprocessor(target_variable='price_direction',
                                          scale_features=False)
    X, y = preprocessor.fit_transform(df, available, 'price_direction')

    # Use first 200-day window for training
    train_window = 200
    X_train = X[:train_window]
    y_train = y[:train_window]

    print(f"\nTraining on first {train_window}-day window")
    print(f"  Features: {X_train.shape[1]}")
    print(f"  Samples: {X_train.shape[0]}")

    # Create model with Optuna params
    model = BayesianClassifier(
        input_size=X_train.shape[1],
        hidden_layers=hidden_layers,
        output_size=2,
        activation=best_params['activation'],
        dropout_rate=best_params['dropout_rate'],
        prior_scale=1.0
    )

    print(f"  Parameters: {model.get_num_parameters()}")

    # Train with snapshot capture
    trainer = SnapshotTrainer(model, device=device,
                               learning_rate=best_params['learning_rate'])

    trainer.fit(
        X_train, y_train,
        batch_size=best_params['batch_size'],
        epochs=best_params['epochs'],
        validation_split=0.1,
        early_stopping_patience=999,
        alpha=best_params['alpha'],
        beta=best_params['beta'],
        verbose=True
    )

    weight_history = trainer.weight_history

    # Generate visualizations
    print(f"\n{'='*60}")
    print("GENERATING VISUALIZATIONS")
    print(f"{'='*60}")

    create_static_plot(weight_history, 'results/weights_final.png')
    create_animation(weight_history, 'results/weights_animation.gif')

    print(f"\nDone! Check results/weights_final.png and results/weights_animation.gif")


if __name__ == "__main__":
    main()
