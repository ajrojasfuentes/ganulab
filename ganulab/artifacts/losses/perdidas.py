# ganulab/artifacts/losses/perdidas.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd

from ganulab.modeling.lossfunction import LossFunction
# Importamos decoradores del init
from . import lossterm, penalty, lossfunc

# =====================================================================
# 1. REGRESSION LOSSES (Continuous Targets)
# =====================================================================

@lossterm()
def mse_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean Squared Error (L2 Norm).
    Robustez: Baja. Penaliza mucho los outliers.
    """
    return F.mse_loss(outputs, labels)

@lossterm()
def l1_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error (L1 Norm).
    Robustez: Alta. Menos sensible a outliers que MSE.
    """
    return F.l1_loss(outputs, labels)

@lossterm()
def smooth_l1_loss(outputs: torch.Tensor, labels: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Huber Loss / Smooth L1.
    Combinación: Comportamiento cuadrático cerca de 0, lineal lejos de 0.
    Ideal para regresión robusta.
    """
    return F.smooth_l1_loss(outputs, labels, beta=beta)

# =====================================================================
# 2. CLASSIFICATION LOSSES (Discrete/Probability Targets)
# =====================================================================

@lossterm()
def bce_distance(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Binary Cross Entropy with Logits.
    Input: Logits (sin sigmoide).
    Target: Probabilidades [0, 1].
    """
    return F.binary_cross_entropy_with_logits(outputs, labels)

@lossterm()
def cross_entropy(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Standard Cross Entropy (Multiclass).
    Input: Logits [B, C].
    Target: Indices de clase [B] (Long) o Probabilidades [B, C] (Float).
    """
    return F.cross_entropy(outputs, labels)

@lossterm()
def nll_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Negative Log Likelihood.
    Input: Log-Probabilidades (LogSoftmax ya aplicado).
    Target: Indices de clase.
    """
    return F.nll_loss(outputs, labels)

@lossterm()
def focal_loss(outputs: torch.Tensor, labels: torch.Tensor, alpha: float = 0.25, gamma: float = 2.0) -> torch.Tensor:
    """
    Focal Loss para clasificación desbalanceada.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    
    Args:
        outputs: Logits [B, C] o [B, 1].
        labels: Targets binarios o índices.
    """
    # Implementación para caso binario/multilabel (BCE base)
    bce_loss = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
    pt = torch.exp(-bce_loss) # prob de la clase correcta
    focal_loss = alpha * (1 - pt) ** gamma * bce_loss
    return focal_loss.mean()

# =====================================================================
# 3. SIMILARITY & EMBEDDING LOSSES
# =====================================================================

@lossterm()
def cosine_loss(outputs: torch.Tensor, labels: torch.Tensor, margin: float = 0.0) -> torch.Tensor:
    """
    Cosine Embedding Loss.
    Mide si dos vectores son similares (labels=1) o disimiles (labels=-1).
    """
    return F.cosine_embedding_loss(outputs, labels, margin=margin)

@lossterm()
def hinge_loss(outputs: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Hinge Loss (SVM).
    Usado comúnmente en SVMs o GANs geométricas.
    Margin default = 1.0.
    """
    return F.hinge_embedding_loss(outputs, labels)

# =====================================================================
# 4. ADVANCED METRICS (GANs / Energy / Distributions)
# =====================================================================

@lossterm()
def wasserstein_distance(real_scores: torch.Tensor | None, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    W-GAN loss / Critic loss.
    """
    if real_scores is None:
        # Generator mode: minimize -E[D(fake)] => maximize E[D(fake)]
        return -fake_scores.mean() 
    else:
        # Critic mode: Maximizar E[D(real)] - E[D(fake)]
        # => Minimizar E[D(fake)] - E[D(real)]
        return fake_scores.mean() - real_scores.mean()

@lossterm()
def energy_distance(real_samples: torch.Tensor, fake_samples: torch.Tensor) -> torch.Tensor:
    """
    Energy Distance (Cramér).
    Implementación robusta V-statistic (incluyendo diagonales).
    D^2(P, Q) = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||
    """
    if real_samples.ndim != 2 or fake_samples.ndim != 2:
        raise ValueError("Samples must be [N, D].")

    # Termino 1: Cruce (X vs Y)
    d_rf = torch.cdist(real_samples, fake_samples, p=2)
    term1 = 2.0 * d_rf.mean()

    # Termino 2: Real vs Real (X vs X')
    d_rr = torch.cdist(real_samples, real_samples, p=2)
    term2 = d_rr.mean()

    # Termino 3: Fake vs Fake (Y vs Y')
    d_ff = torch.cdist(fake_samples, fake_samples, p=2)
    term3 = d_ff.mean()

    return term1 - term2 - term3

# =====================================================================
# 5. PENALTIES (Regularization & Structural)
# =====================================================================

@penalty()
def l1_regularization(module: nn.Module) -> torch.Tensor:
    """
    Penalización L1 (Lasso) sobre todos los parámetros.
    Promueve la escasez (sparsity) en los pesos.
    """
    loss = torch.tensor(0.0, device=next(module.parameters()).device)
    for param in module.parameters():
        loss = loss + param.abs().sum()
    return loss

@penalty()
def l2_regularization(module: nn.Module) -> torch.Tensor:
    """
    Penalización L2 (Ridge / Weight Decay).
    Evita que los pesos crezcan demasiado.
    Nota: Los optimizadores suelen tener 'weight_decay', pero esto permite 
    aplicarlo como parte explícita de la LossFunction.
    """
    loss = torch.tensor(0.0, device=next(module.parameters()).device)
    for param in module.parameters():
        loss = loss + param.pow(2).sum()
    return loss

@penalty()
def kl_loss(module: nn.Module) -> torch.Tensor:
    """
    Suma de KL Divergence (Redes Bayesianas).
    Busca el método .kl_loss() en el módulo.
    """
    if hasattr(module, 'kl_loss'):
        return module.kl_loss()
    return torch.tensor(0.0, device=next(module.parameters()).device)

@penalty()
def width_loss(module: nn.Module, p: float = 2.0) -> torch.Tensor:
    """
    Suma de Width Loss (Redes Intervalares).
    Busca el método .width_loss() en el módulo.
    """
    if hasattr(module, 'width_loss'):
        return module.width_loss(p=p)
    return torch.tensor(0.0, device=next(module.parameters()).device)

@penalty()
def gradient_penalty(module: nn.Module, inputs: torch.Tensor, target_norm: float = 1.0) -> torch.Tensor:
    """
    WGAN-GP Gradient Penalty: E[(||grad||_2 - 1)^2]
    Asegura la restricción de 1-Lipschitz en el discriminador.
    """
    x_hat = inputs.clone().detach().requires_grad_(True)
    pred = module(x_hat)

    grad_outputs = torch.ones_like(pred)
    
    gradients = autograd.grad(
        outputs=pred,
        inputs=x_hat,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    grad_norm = gradients.norm(2, dim=1)
    
    gp = ((grad_norm - target_norm) ** 2).mean()
    return gp

# =====================================================================
# 6. PRE-CONFIGURED LOSS FUNCTIONS (Recipes / Presets)
# =====================================================================

@lossfunc()
def wgan_gp_discriminator(lambda_gp: float = 10.0) -> "LossFunction":
    """
    Receta estándar para entrenar el Discriminador (Crítico) en WGAN-GP.
    Combina:
      - Wasserstein Distance (Maximizar gap entre real y fake).
      - Gradient Penalty (Forzar restricción 1-Lipschitz).
    
    Uso:
        criterion = lflib.lossfunc.wgan_gp_discriminator()
        loss = criterion(real_scores=r, fake_scores=f, module=D, inputs=interpolated)
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, wasserstein_distance) \
        .add_penalty(lambda_gp, gradient_penalty)

@lossfunc()
def wgan_generator() -> "LossFunction":
    """
    Receta estándar para el Generador en WGAN.
    Solo busca maximizar la puntuación que el crítico da a los fakes.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, wasserstein_distance)

@lossfunc()
def energy_matching(lambda_reg: float = 1e-4) -> "LossFunction":
    """
    Receta para 'Feature Matching' o Generación basada en Energía.
    Combina:
      - Energy Distance (Minimizar discrepancia estadística).
      - L2 Regularization (Evitar explosión de pesos en el generador).
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, energy_distance) \
        .add_penalty(lambda_reg, l2_regularization)

@lossfunc()
def variational_elbo(beta: float = 0.1, task_loss: str = "mse") -> "LossFunction":
    """
    ELBO (Evidence Lower Bound) para Redes Bayesianas.
    Combina:
      - Task Loss (MSE o BCE) -> Likelihood.
      - KL Divergence -> Prior matching.
    
    Args:
        beta: Peso de la divergencia KL (trade-off complejidad/ajuste).
        task_loss: 'mse' (regresión) o 'bce' (clasificación).
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    lf = LossFunction()
    
    if task_loss.lower() == "mse":
        lf.set_main(1.0, mse_loss)
    elif task_loss.lower() == "bce":
        lf.set_main(1.0, bce_distance)
    else:
        raise ValueError(f"Task loss '{task_loss}' no soportada en preset ELBO.")
        
    return lf.add_penalty(beta, kl_loss)

@lossfunc()
def robust_interval_regression(lambda_width: float = 0.01) -> "LossFunction":
    """
    Receta para Regresión Intervalar Robusta.
    Combina:
      - Smooth L1 (Huber) -> Ajuste central robusto a outliers.
      - Width Loss -> Minimizar la incertidumbre (ancho del intervalo).
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, smooth_l1_loss) \
        .add_penalty(lambda_width, width_loss)

@lossfunc()
def cramer_critic(lambda_gp: float = 10.0) -> "LossFunction":
    """
    Receta pulida para el Crítico en una Cramer GAN Estándar.
    
    Objetivo:
      - Maximizar la Energy Distance entre características Reales y Falsas.
        (Por eso usamos peso -1.0 en main, para minimizar el negativo).
      - Mantener restricción Lipschitz mediante Gradient Penalty.
    
    Args:
        lambda_gp: Peso de la penalización de gradiente (default 10.0).
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    # Loss = -1 * Energy_Distance + 10 * GP
    return LossFunction() \
        .set_main(-1.0, energy_distance) \
        .add_penalty(lambda_gp, gradient_penalty)

@lossfunc()
def bayesian_cramer_critic(lambda_gp: float = 10.0, beta: float = 0.01) -> "LossFunction":
    """
    Receta para el Crítico en una Cramer GAN Bayesiana.
    
    Combina:
      - Geometría de Cramer (Energy Distance + GP).
      - Incertidumbre Bayesiana (KL Divergence).
      
    Utilidad:
      Ayuda a evitar el 'mode collapse' al permitir que el crítico tenga
      incertidumbre sobre regiones del espacio no exploradas.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(-1.0, energy_distance) \
        .add_penalty(lambda_gp, gradient_penalty) \
        .add_penalty(beta, kl_loss)

@lossfunc()
def interval_cramer_critic(lambda_gp: float = 10.0, lambda_width: float = 0.01) -> "LossFunction":
    """
    Receta para el Crítico en una Cramer GAN Intervalar (Robusta).
    
    Combina:
      - Geometría de Cramer (Energy Distance + GP).
      - Robustez Intervalar (Width Loss).
      
    Utilidad:
      Entrena un crítico que no solo distingue real/fake, sino que garantiza
      cotas de seguridad en su predicción, haciendo el entrenamiento más estable
      ante ruido adversarial.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(-1.0, energy_distance) \
        .add_penalty(lambda_gp, gradient_penalty) \
        .add_penalty(lambda_width, width_loss)

@lossfunc()
def cramer_generator() -> "LossFunction":
    """
    Receta estándar para el Generador en una Cramer GAN.
    
    Objetivo:
      - Minimizar la Energy Distance entre las muestras generadas y las reales.
      - (Hacer que la distribución Fake converja a la Real).
    
    Nota de uso:
      A diferencia de WGAN clásica, la Energy Distance requiere pasar
      tanto 'real_samples' como 'fake_samples' al calcular el loss del generador.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    # Peso 1.0 positivo => Minimizar distancia
    return LossFunction() \
        .set_main(1.0, energy_distance)

@lossfunc()
def bayesian_cramer_generator(beta: float = 0.001) -> "LossFunction":
    """
    Receta para un Generador Bayesiano en Cramer GAN.
    
    Combina:
      - Minimización de Energy Distance (Calidad de imagen).
      - KL Divergence (Regularización de pesos).
    
    Utilidad:
      Evita el sobreajuste del generador a modos específicos (Mode Collapse)
      manteniendo diversidad en los pesos estocásticos.
      Se suele usar un beta bajo para no sacrificar demasiada calidad visual.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, energy_distance) \
        .add_penalty(beta, kl_loss)

@lossfunc()
def interval_cramer_generator(lambda_width: float = 0.01) -> "LossFunction":
    """
    Receta para un Generador Intervalar en Cramer GAN.
    
    Combina:
      - Minimización de Energy Distance.
      - Width Loss (Control de incertidumbre).
    
    Utilidad:
      Genera muestras robustas. Al penalizar el ancho (width), forzamos al
      generador a ser "decisivo" en sus píxeles, evitando imágenes grises o borrosas
      causadas por intervalos de salida demasiado amplios.
    """
    from ganulab.modeling.lossfunction import LossFunction
    
    return LossFunction() \
        .set_main(1.0, energy_distance) \
        .add_penalty(lambda_width, width_loss)