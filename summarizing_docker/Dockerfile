# Usar una imagen base de Python 3.11 
FROM python:3.10.12


# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copiar requirements.txt desde la carpeta /app e instalar dependencias
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar todo el contenido del proyecto, incluyendo la carpeta /app
COPY . .

# Exponer el puerto que usará la aplicación (si es una web, por ejemplo)
EXPOSE 8000

# Ejecutar el script main.py con uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]