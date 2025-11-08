import pandas as pd
import numpy as np

class PreprocesamientoDatos:
    """Clase para el preprocesamiento y discretización de datos"""
    
    def __init__(self):
        """Inicializar el preprocesador con los bins definidos en la tarea"""
        # Bins para Edad (3 categorías |  Joven, Adulto, Mayor)
        self.bins_edad = [17, 29, 59, 75]
        self.etiquetas_edad = [0, 1, 2]  # Joven, Adulto, Mayor
        self.nombres_edad = ['Joven (18-29)', 'Adulto (30-59)', 'Mayor (60-75)']
        
        # Bins para Ingreso (5 categorías | Muy bajo, Bajo, Medio, Alto, Muy alto)
        self.bins_ingreso = [9999, 19999, 29999, 39999, 49999, 60000]
        self.etiquetas_ingreso = [0, 1, 2, 3, 4]  # 0 = Muy bajo, 1 = Bajo, 2 = Medio, 3 = Alto, 4 = Muy alto
        self.nombres_ingreso = [
            'Muy bajo (10k-19,999k)',
            'Bajo (20k-29,999k)',
            'Medio (30k-39,999k)',
            'Alto (40k-49,999k)',
            'Muy alto (50k-60k)'
        ]
    
    def aplicar_binning(self, df):
        """Aplicar discretización a las variables Edad e Ingreso"""
        df_procesado = df.copy()

        # Aplicar binning a Edad
        df_procesado['Edad_bin'] = pd.cut(
            df_procesado['Edad'],
            bins=self.bins_edad,
            labels=self.etiquetas_edad,
            include_lowest=True
        )

        # Aplicar binning a Ingreso
        df_procesado['Ingreso_bin'] = pd.cut(
            df_procesado['Ingreso'],
            bins=self.bins_ingreso,
            labels=self.etiquetas_ingreso,
            include_lowest=True
        )

        # conversion a enteros para evitar problemas con categorias
        df_procesado['Edad_bin'] = df_procesado['Edad_bin'].astype(int)
        df_procesado['Ingreso_bin'] = df_procesado['Ingreso_bin'].astype(int)

        return df_procesado

    def obtener_descripcion_bins(self):
        """Obtener descripción de los bins utilizados"""
        return {
            'edad': {
                'bins': self.bins_edad,
                'etiquetas': self.etiquetas_edad,
                'nombres': self.nombres_edad
            },
            'ingreso': {
                'bins': self.bins_ingreso,
                'etiquetas': self.etiquetas_ingreso,
                'nombres': self.nombres_ingreso
            }
        }

    def mostrar_estadisticas_binning(self, df_original, df_procesado):
        """Mostrar estadísticas del proceso de binning"""

        print("=== ESTADÍSTICAS DEL BINNING ===")

        print("\n--- DISTRIBUCIÓN POR BINS DE EDAD ---")
        distribucion_edad = df_procesado['Edad_bin'].value_counts().sort_index()
        for bin_idx, count in distribucion_edad.items():
            nombre = self.nombres_edad[bin_idx]
            porcentaje = (count / len(df_procesado)) * 100
            print(f"Bin {bin_idx} - {nombre}: {count} personas ({porcentaje:.1f}%)")

        print("\n--- DISTRIBUCIÓN POR BINS DE INGRESO ---")
        distribucion_ingreso = df_procesado['Ingreso_bin'].value_counts().sort_index()
        for bin_idx, count in distribucion_ingreso.items():
            nombre = self.nombres_ingreso[bin_idx]
            porcentaje = (count / len(df_procesado)) * 100
            print(f"Bin {bin_idx} - {nombre}: {count} personas ({porcentaje:.1f}%)")
        
        print("\n--- TABLA CRUZADA: EDAD_BIN vs INGRESO_BIN ---")
        tabla_cruzada = pd.crosstab(df_procesado['Edad_bin'], df_procesado['Ingreso_bin'])
        print(tabla_cruzada)
    
    def convertir_valores_a_bins(self, edad, ingreso):
        """Convertir valores individuales de edad e ingreso a sus bins correspondientes"""

        # Determinar bin de rango de edades
        if edad <= 29:
            edad_bin = 0
        elif edad <= 59:
            edad_bin = 1
        else:
            edad_bin = 2

        # Determinar bin de rango de ingresos
        if ingreso < 19999:
            ingreso_bin = 0
        elif ingreso < 29999:
            ingreso_bin = 1
        elif ingreso < 39999:
            ingreso_bin = 2
        elif ingreso < 49999:
            ingreso_bin = 3
        else:
            ingreso_bin = 4
        
        return edad_bin, ingreso_bin
