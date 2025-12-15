"""
GANU-Lab Training Engine.

This module provides a universal 'Trainer' class capable of handling:
1. Standard Supervised/Unsupervised Learning (Single Model).
2. Adversarial Learning (GANs - Generator & Discriminator).

It integrates seamlessly with NeuralModel, LossFunction, IO, and Display utilities.
"""

import os
import time
from pathlib import Path
from typing import Optional, Union, Dict, List, Any, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

# Optional tqdm for progress bars
try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(x, **kwargs): return x

# Framework Imports
from ganulab.modeling.neuralmodel import NeuralModel
from ganulab.utils import io, display

class Trainer:
    """
    Universal training engine for GANU-Lab models.
    
    Modes:
        - STANDARD: Trains a single NeuralModel (Regression, Classification, Autoencoders).
        - GAN: Trains a Generator and Discriminator pair (WGAN, CramerGAN, etc.).
    """
    
    def __init__(
        self,
        model: Union[NeuralModel, Dict[str, NeuralModel]],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        use_amp: bool = True,
        save_dir: str = "checkpoints"
    ):
        """
        Args:
            model: A single NeuralModel or a dict {'G': model, 'D': model}.
            device: Target device for execution.
            use_amp: Enable Automatic Mixed Precision (FP16) for speed/memory.
            save_dir: Directory to save checkpoints and logs.
        """
        self.device = device
        self.use_amp = use_amp and (device != "cpu")
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # --- Mode Detection ---
        if isinstance(model, dict):
            if 'G' not in model or 'D' not in model:
                raise ValueError("For GAN mode, dict must contain keys 'G' and 'D'.")
            self.mode = "GAN"
            self.models = model
            # Ensure models are on the correct device
            self.models['G'].to(device)
            self.models['D'].to(device)
            print(f"[Trainer] Mode: GAN (Adversarial)")
        
        elif isinstance(model, NeuralModel):
            self.mode = "STANDARD"
            self.model = model
            self.model.to(device)
            print(f"[Trainer] Mode: STANDARD (Supervised/Unsup)")
        
        else:
            raise TypeError("Model must be a NeuralModel or a dict {'G':..., 'D':...}")

        # AMP Scaler
        self.scaler = GradScaler(enabled=self.use_amp)
        
        # Metrics History
        self.history: Dict[str, List[float]] = {}

    # ==========================================================================
    # 1. STEP LOGIC
    # ==========================================================================

    def _step_standard(self, batch: Any) -> Dict[str, float]:
        """Single training step for standard models."""
        model = self.model
        optimizer = model.optimizer
        loss_fn = model.loss_fn

        if loss_fn is None:
            raise RuntimeError("Model has no loss_fn configured.")

        # Unpack batch (Features, Targets)
        # NeuralDataset yields: x or (x, y)
        x, y = None, None
        if isinstance(batch, (list, tuple)):
            x = batch[0].to(self.device)
            if len(batch) > 1: 
                y = batch[1].to(self.device)
        else:
            x = batch.to(self.device)

        optimizer.zero_grad()

        # Mixed Precision Context
        with autocast(enabled=self.use_amp):
            # Forward
            y_pred = model(x)
            
            # Loss Calculation (Automatic introspection of arguments)
            loss = loss_fn(
                outputs=y_pred, 
                labels=y, 
                inputs=x,       # Useful for autoencoders (input reconstruction)
                module=model    # Useful for KL/Width penalties
            )

        # Backward & Step
        self.scaler.scale(loss).backward()
        self.scaler.step(optimizer)
        self.scaler.update()

        return {"loss": loss.item()}

    def _step_gan(self, batch: Any) -> Dict[str, float]:
        """Single training step for GANs."""
        netG = self.models['G']
        netD = self.models['D']
        optG = netG.optimizer
        optD = netD.optimizer
        
        # Prepare Real Data
        if isinstance(batch, (list, tuple)):
            real_data = batch[0].to(self.device)
        else:
            real_data = batch.to(self.device)
            
        batch_size = real_data.size(0)

        # ------------------------------------------------------------------
        # (1) Update Discriminator (Critic)
        # ------------------------------------------------------------------
        optD.zero_grad()
        
        with autocast(enabled=self.use_amp):
            # Generate Fake Data (Detached from G graph)
            # NeuralModel.forward(batch_size=N) uses the internal LatentSampler
            fake_data = netG(batch_size=batch_size).detach()
            
            # Scores
            d_real = netD(real_data)
            d_fake = netD(fake_data)
            
            # Gradient Penalty Interpolation (On-the-fly)
            # Needed for WGAN-GP / Cramer GAN stability
            alpha = torch.rand(batch_size, 1, device=self.device)
            # Broadcast alpha to match data dims (assuming [B, Features...])
            for _ in range(real_data.ndim - 2): 
                alpha = alpha.unsqueeze(-1)
            
            interpolated = (alpha * real_data + (1 - alpha) * fake_data)
            interpolated.requires_grad_(True)
            
            # Loss D calculation
            # 'inputs' argument is passed for gradient_penalty inspection
            loss_d = netD.loss_fn(
                real_scores=d_real, 
                fake_scores=d_fake, 
                real_samples=real_data, 
                fake_samples=fake_data,
                module=netD,
                inputs=interpolated 
            )

        self.scaler.scale(loss_d).backward()
        self.scaler.step(optD)
        
        # ------------------------------------------------------------------
        # (2) Update Generator
        # ------------------------------------------------------------------
        optG.zero_grad()
        
        with autocast(enabled=self.use_amp):
            # Generate Fake Data (With gradients for G)
            fake_data_g = netG(batch_size=batch_size)
            
            # Evaluate with D (Frozen conceptually)
            d_fake_g = netD(fake_data_g)
            
            # Loss G
            # real_scores=None signals Generator mode to loss functions like Wasserstein
            loss_g = netG.loss_fn(
                real_scores=None, 
                fake_scores=d_fake_g,
                real_samples=real_data,   # Needed for Energy Distance
                fake_samples=fake_data_g,
                module=netG
            )

        self.scaler.scale(loss_g).backward()
        self.scaler.step(optG)
        self.scaler.update()

        return {"loss_D": loss_d.item(), "loss_G": loss_g.item()}

    # ==========================================================================
    # 2. MAIN LOOP
    # ==========================================================================

    def fit(
        self, 
        train_loader: DataLoader, 
        epochs: int, 
        val_loader: Optional[DataLoader] = None,
        checkpoint_interval: int = 10,
        plot_interval: int = 5
    ):
        """
        Runs the training loop.
        
        Args:
            train_loader: DataLoader for training data.
            epochs: Number of epochs.
            val_loader: DataLoader for validation (Standard mode only).
            checkpoint_interval: Save model every N epochs.
            plot_interval: Save history plot every N epochs.
        """
        print(f"\n🚀 Starting training for {epochs} epochs on {self.device}...")
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            epoch_loss = {}
            steps = 0
            
            # --- Set Train Mode ---
            if self.mode == "STANDARD": 
                self.model.train()
            else: 
                self.models['G'].train()
                self.models['D'].train()

            # Progress Bar
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False) if HAS_TQDM else train_loader
            
            for batch in pbar:
                steps += 1
                
                # Execute Step
                if self.mode == "STANDARD":
                    metrics = self._step_standard(batch)
                else:
                    metrics = self._step_gan(batch)
                
                # Accumulate
                for k, v in metrics.items():
                    epoch_loss[k] = epoch_loss.get(k, 0.0) + v
                
                # Update Pbar (Show main loss)
                if HAS_TQDM:
                    main_key = list(metrics.keys())[0]
                    pbar.set_postfix({main_key: f"{metrics[main_key]:.4f}"})

            # Average metrics
            for k in epoch_loss:
                epoch_loss[k] /= steps
                
                # Append to History
                if k not in self.history: self.history[k] = []
                self.history[k].append(epoch_loss[k])

            # --- Validation (Standard Mode Only) ---
            log_parts = [f"Epoch {epoch}"]
            for k, v in epoch_loss.items():
                log_parts.append(f"{k}: {v:.4f}")

            if self.mode == "STANDARD" and val_loader is not None:
                val_metrics = self._validate(val_loader)
                for k, v in val_metrics.items():
                    if k not in self.history: self.history[k] = []
                    self.history[k].append(v)
                    log_parts.append(f"{k}: {v:.4f}")
            
            print(" | ".join(log_parts))

            # --- Checkpointing ---
            if epoch % checkpoint_interval == 0 or epoch == epochs:
                self._save_checkpoints(epoch, epoch_loss)

            # --- Live Plotting (Save to disk) ---
            if epoch % plot_interval == 0:
                display.plot_training_runs(
                    self.history, 
                    title=f"Training History (Epoch {epoch})",
                    save_path=str(self.save_dir / "history_live.png")
                )

        total_time = time.time() - start_time
        print(f"\n✅ Training finished in {total_time/60:.2f} minutes.")
        
        # Final Plot
        display.plot_training_runs(self.history, title="Final Training Metrics")

    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validation loop for Standard mode."""
        self.model.eval()
        val_loss = 0.0
        steps = 0
        
        with torch.no_grad():
            for batch in val_loader:
                steps += 1
                x, y = None, None
                if isinstance(batch, (list, tuple)):
                    x = batch[0].to(self.device)
                    if len(batch) > 1: y = batch[1].to(self.device)
                else:
                    x = batch.to(self.device)
                
                y_pred = self.model(x)
                # Compute loss
                loss = self.model.loss_fn(outputs=y_pred, labels=y, module=self.model)
                val_loss += loss.item()
        
        return {"val_loss": val_loss / max(steps, 1)}

    def _save_checkpoints(self, epoch: int, metrics: Dict[str, float]):
        """Delegates saving to ganulab.utils.io."""
        if self.mode == "STANDARD":
            io.save_model(
                self.model, 
                self.save_dir, 
                filename=f"model_epoch_{epoch}", 
                metadata={"epoch": epoch, **metrics},
                verbose=False
            )
        else:
            io.save_model(
                self.models['G'], 
                self.save_dir, 
                filename=f"generator_epoch_{epoch}", 
                metadata={"epoch": epoch, **metrics},
                verbose=False
            )
            io.save_model(
                self.models['D'], 
                self.save_dir, 
                filename=f"discriminator_epoch_{epoch}", 
                metadata={"epoch": epoch, **metrics},
                verbose=False
            )