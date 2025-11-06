import pandas as pd
import numpy as np

class ClasificadorBayesiano:
    """Clasificador bayesiano implementado manualmente usando el Teorema de Bayes"""

    def __init__(self):
        """Inicializar el clasificador"""
        # Diccionario para probabilidades a priori P(C)
        self.probabilidades_priori = {}

        # Variables para entrenamiento del bayesiano
        self.datos_entrenamiento = None
        self.clases_unicas = None
        self.total_muestras = 0

        print("Clasificador bayesiano cargado y listo para entrenar.")

    def calcular_probabilidades_priori(self, df_entrenamiento):
        """Calcular las probabilidades a priori "P(C)" para cada clase (Camina, Conduce)
        P(C) = frecuencia_de_clase / total_de_muestras_clases """
        print("=== CALCULANDO PROBABILIDADES A PRIORI P(C) ===")

        # Guardar datos de entrenamiento
        self.datos_entrenamiento = df_entrenamiento
        self.total_muestras = len(df_entrenamiento)

        # Obtener clases únicas
        self.clases_unicas = df_entrenamiento['Clase'].unique()
        print(f"Mostrando clases: {self.clases_unicas}")

        # contar cuantas muestras hay de cada clase
        conteo_clases = df_entrenamiento['Clase'].value_counts()
        print(f"\nConteo por clase:")
        for clase, count in conteo_clases.items():
            print(f"  {clase}: {count} muestras")

        # Calcular probabilidades a priori
        print(f"\nProbabilidades a priori P(C):")
        for clase in self.clases_unicas:
            frecuencia = conteo_clases[clase]
            # P(C) = frecuencia_de_clase / total_de_muestras_clases
            probabilidad = frecuencia / self.total_muestras
            # Guardar en el diccionario
            self.probabilidades_priori[clase] = probabilidad

            print(f"  P({clase}) = {frecuencia}/{self.total_muestras} = {probabilidad:.4f}")

        return self.probabilidades_priori

    def mostrar_probabilidades_priori(self):
        """Mostrar las probabilidades a priori calculadas"""

        if not self.probabilidades_priori:
            print("Error: No se han calculado las probabilidades a priori aun")
            return

        print("\n=== RESUMEN DE PROBABILIDADES A PRIORI ===")
        for clase, probabilidad in self.probabilidades_priori.items():
            porcentaje = probabilidad * 100
            print(f"P({clase}) = {probabilidad:.4f} ({porcentaje:.1f}%)")

    def calcular_probabilidades_conjuntas(self, df_entrenamiento):
        # 2.2 Calcular la probabilidad conjunta P(Edad_bin, Ingreso_bin | Clase)
        """Calcular las probabilidades conjuntas P(Edad_bin, Ingreso_bin | Clase)

        probabilidad de que una persona tenga una combinación específica de edad e ingreso
        dado que pertenece a una clase determinada.

        P(Edad_bin=i, Ingreso_bin=j | Clase=c) = frecuencia_de_combinacion(i,j,c) / total_de_clase_c"""

        print("=== CALCULANDO PROBABILIDADES CONJUNTAS P(x|C) ===")

        # Diccionario para guardar todas las probabilidades conjuntas
        self.probabilidades_conjuntas = {}

        for clase in self.clases_unicas:
            print(f"\n--- Clase: {clase} ---")

            # Filtrar los datos de esta clase
            datos_clase = df_entrenamiento[df_entrenamiento['Clase'] == clase]
            total_clase = len(datos_clase)

            print(f"Total de muestras en clase '{clase}': {total_clase}")

            # Crear tabla de frecuencias cruzadas para la clase
            # a) Agrupa los datos por combinacion de bins y clases
            tabla_frecuencias = datos_clase.groupby(['Edad_bin', 'Ingreso_bin']).size().reset_index(name='Frecuencia')

            print(f"Combinaciones encontradas: {len(tabla_frecuencias)}")

            # Calcular probabilidades para cada combinación
            self.probabilidades_conjuntas[clase] = {}

            # Se itera sobre cada fila de la tabla de frecuencias
            for _, fila in tabla_frecuencias.iterrows():
                edad_bin = fila['Edad_bin']
                ingreso_bin = fila['Ingreso_bin']
                frecuencia = fila['Frecuencia']

                # Calcular P(Edad_bin, Ingreso_bin | Clase)
                probabilidad = frecuencia / total_clase

                # Tupla como clave para la combinación
                combinacion = (edad_bin, ingreso_bin)
                self.probabilidades_conjuntas[clase][combinacion] = probabilidad

                # P(Edad_bin, Ingreso_bin | Clase) = frecuencia_de_combinacion(i,j,c) / total_de_clase_c = probabilidad
                print(f"  P(Edad_bin={edad_bin}, Ingreso_bin={ingreso_bin} | {clase}) = {frecuencia}/{total_clase} = {probabilidad:.4f}")

        return self.probabilidades_conjuntas

    def mostrar_tabla_probabilidades_conjuntas(self):
        """Mostrar tabla organizada de probabilidades conjuntas"""

        # hasattr = verifica si el objeto tiene un atributo específico (booleano)
        if not hasattr(self, 'probabilidades_conjuntas'):
            print("Error: No se han calculado las probabilidades conjuntas aún")
            return

        print("\n=== TABLA DE PROBABILIDADES CONJUNTAS ===")
        print("Formato: P(Edad_bin, Ingreso_bin | Clase)")

        for clase in self.clases_unicas:
            print(f"\n--- Clase: {clase} ---")
            print("Edad_bin\tIngreso_bin\tProbabilidad")
            print("-" * 40)

            for (edad_bin, ingreso_bin), probabilidad in self.probabilidades_conjuntas[clase].items():
                print(f"{edad_bin}\t\t{ingreso_bin}\t\t{probabilidad:.4f}")

    def obtener_probabilidad_conjunta(self, edad_bin, ingreso_bin, clase):
        """Obtener probabilidad conjunta específica para una combinación """

        if not hasattr(self, 'probabilidades_conjuntas'):
            print("Error: No se han calculado las probabilidades conjuntas aún")
            return 0

        combinacion = (edad_bin, ingreso_bin)

        # Retorna la probabilidad conjunta o 0 si no existe
        return self.probabilidades_conjuntas[clase].get(combinacion, 0)

    def calcular_probabilidad_evidencia(self, edad_bin, ingreso_bin):
        """ Metodo para calcular la probabilidad de la evidencia P(x) para una combinación de bins.
        P(x) = sum_c_{camina, conduce} [ P(x|C) * P(C) ]
        """
        # Verificar que las probabilidades a priori y conjuntas estén calculadas
        if not self.probabilidades_priori or not hasattr(self, 'probabilidades_conjuntas'):
            print("Error: Primero debes calcular las probabilidades a priori y conjuntas.")
            return 0

        # Inicializar P(x)
        px = 0

        print(f"\n=== CALCULO DE EVIDENCIA P(x) PARA (Edad_bin={edad_bin}, Ingreso_bin={ingreso_bin}) ===")
        # Iterar sobre cada clase para calcular la suma
        for clase in self.clases_unicas:
            # Obtener P(x|C) y P(C)
            # P(x|C) = P(Edad_bin, Ingreso_bin | Clase)
            p_x_dado_c = self.obtener_probabilidad_conjunta(edad_bin, ingreso_bin, clase)
            # P(C) = probabilidad a priori de la clase
            p_c = self.probabilidades_priori[clase]
            # Sumar al total P(x) = P(x|C) * P(C)
            producto = p_x_dado_c * p_c
            px += producto
            print(f"  P(x|{clase}) = {p_x_dado_c:.4f}  |  P({clase}) = {p_c:.4f}  |  Producto = {producto:.4f}")
        print(f"\n  P(x) = {px:.4f}\n")
        return px

    def predecir_clase(self, edad_bin, ingreso_bin):
        """Predecir la clase (Camina o Conduce) para combinación de bins usando el Teorema de Bayes
        P(C|x) = [ P(x|C) * P(C) ] / P(x)
        """

        print(f"\n=== PREDICCIÓN DE CLASE PARA (Edad_bin={edad_bin}, Ingreso_bin={ingreso_bin}) ===")

        # Calcular la evidencia P(x)
        p_x = self.calcular_probabilidad_evidencia(edad_bin, ingreso_bin)

        # Diccionario = guardar las probabilidades posteriores P(C|x)
        probabilidades_posteriores = {}

        # Calcular P(C|x) para cada clase
        for clase in self.clases_unicas:
            # Obtener P(x|C) y P(C)
            p_x_dado_c = self.obtener_probabilidad_conjunta(edad_bin, ingreso_bin, clase)
            p_c = self.probabilidades_priori[clase]

            # Aplicar el Teorema de Bayes
            if p_x > 0:
                # P(C|x) = [ P(x|C) * P(C) ] / P(x)
                p_c_dado_x = (p_x_dado_c * p_c) / p_x
            else:
                p_c_dado_x = 0

            # Guardar la probabilidad posterior
            probabilidades_posteriores[clase] = p_c_dado_x

            print(f"  P({clase}|x) = [ {p_x_dado_c:.4f} * {p_c:.4f} ] / {p_x:.4f} = {p_c_dado_x:.4f}")

        # Determinar la clase con la mayor probabilidad posterior
        clase_predicha = max(probabilidades_posteriores, key=probabilidades_posteriores.get)
        probabilidad_maxima = probabilidades_posteriores[clase_predicha]

        print(f"\nClase predicha: {clase_predicha} con probabilidad {probabilidad_maxima:.4f}\n")

        return clase_predicha, probabilidad_maxima
