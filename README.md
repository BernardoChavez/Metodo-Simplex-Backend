# Metodo Simplex Backend — Simplex FICCT (IO1)

Backend en **FastAPI (Python)** para el cálculo y resolución del **Método Simplex de Dos Fases**, operaciones de fila de Gauss-Jordan, análisis de sensibilidad post-óptima y formulación del problema dual.

Desarrollado para la materia de **Investigación Operativa I** de la **Facultad Integral de Ciencias de la Computación y Telecomunicaciones (FICCT - UAGRM)**.

---

## 🚀 Características
- **Método Simplex de Dos Fases**: Soporte completo para restricciones mixtas ($\le$, $\ge$, $=$) mediante adición de holguras, excesos y variables artificiales.
- **Pivoteo Gauss-Jordan Paso a Paso**: Generación de matrices y registro detallado de las operaciones elementales de fila en cada iteración.
- **Análisis de Sensibilidad**: Intervalos de variación permisibles para los costos unitarios ($c_j$) y lados derechos ($b_i$).
- **Problema Dual y Precios Sombra**: Matriz inversa de la base $B^{-1}$ y cálculo de multiplicadores duales ($y = c_B B^{-1}$).
- **Detección de Casos Especiales**: Óptimos alternativos (múltiples soluciones), no acotamiento e infactibilidad.

---

## 🛠️ Instalación y Ejecución

```bash
# 1. Crear y activar entorno virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Iniciar servidor FastAPI
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Endpoint:** `http://localhost:8000/api/v1/simplex/solve`
- **Swagger Docs:** `http://localhost:8000/docs`

---

## 🧪 Pruebas Unitarias
```bash
PYTHONPATH=. python3 -m unittest discover -s tests
```

