def hola_desde_framework():
    """Función original que saluda desde el framework"""
    return "¡Hola desde el framework!"

def hola_de_nuevo():
    """Nueva función que saluda nuevamente"""
    return "¡Hola desde el framework de nuevo!"

def saludo_personalizado(nombre: str):
    """Saludo personalizado usando PyTorch para demostrar uso"""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return f"¡Hola {nombre}! PyTorch está usando: {device}"