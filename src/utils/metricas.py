"""
Módulo de métricas para evaluación de clasificadores
Funciones: sensibilidad, especificidad y F1-score
Todas las funciones tienen comentarios en español.
"""

def sensibilidad(VP, FN):
    """
    Calcula la sensibilidad (recall) del modelo.
    VP: Verdaderos positivos
    FN: Falsos negativos
    """
    if (VP + FN) == 0:
        return 0
    return VP / (VP + FN)

def especificidad(VN, FP):
    """
    Calcula la especificidad del modelo.
    VN: Verdaderos negativos
    FP: Falsos positivos
    """
    if (VN + FP) == 0:
        return 0
    return VN / (VN + FP)
