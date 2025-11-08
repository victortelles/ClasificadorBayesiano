# Clasificador Bayesiano Manual

Este proyecto implementa un clasificador bayesiano desde cero para predecir si una persona camina o conduce, utilizando datos de edad e ingreso. Todo el procesamiento y la lógica se realiza manualmente, sin librerías externas de machine learning.

## Requisitos
- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## Instalación del entorno virtual
1. Abre una terminal en la carpeta raíz del proyecto.
2. Crea un entorno virtual:
   ```
   python -m venv venv
   ```
3. Activa el entorno virtual:
   - En Windows:
     ```
     venv\Scripts\activate
     ```
   - En Linux/Mac:
     ```
     source venv/bin/activate
     ```
4. Instala las dependencias del proyecto:
   ```
   pip install -r requirements.txt
   ```

## Ejecución del Jupyter Notebook
1. Con el entorno virtual activado, inicia Jupyter Notebook:
   ```
   jupyter notebook
   ```
2. Se abrirá una ventana en el navegador. Navega hasta el archivo `clasificador_bayesiano.ipynb` y ábrelo.
3. Ejecuta las celdas del notebook en orden para reproducir el análisis y los resultados.

## Estructura del proyecto
- `src/` : Código fuente (modelos, preprocesamiento, utilidades)
- `data/` : Dataset utilizado
- `clasificador_bayesiano.ipynb` : Notebook principal con el procedimiento paso a paso
- `requirements.txt` : Dependencias necesarias

