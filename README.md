# CNNDeteccionDeIncendios

Red neuronal convolucional que se encarga de analizar y clasificar imágenes de los distintos lugares de una vivienda, de tal manera que se logre detectar patrones que indiquen un posible incendio.

# Preparación

1. Instale:

- tensorflow
- pathlib
- numpy
- matplotlib

2. Descargue y descomprima el zip con las imagenes para el entrenamiento del modelo:

```
https://uniprivado.s3.amazonaws.com/imagenes_modelo.zip
```

3. Remplace la linea 17 del modelo por la ruta donde se encuentra ubicada la carpeta de imagenes:

```
data_dir = pathlib.Path("D:\Sistema\Descargas\imagenes_modelo")
```

# Creación y entrenamiento del modelo

Ejecute el archivo RNNTony.py para que se cree y entrene el modelo. Al finalizar se mostrará los gráficos de presición del modelo y se guardará en la carpeta models con el nombre de modelTony.h5

# Test del modelo

En el archivo Test.py cambiar la url de la imagen, por la que desea testear en la linea 13:

```
foto_url = "URL_IMAGEN"
```

Y en la linea 14 colocar un nombre distintivo:

```
foto_path = tf.keras.utils.get_file('NOMBRE_IMAGEN', origin=foto_url)
```

Ejecutar el archivo y esperar el resultado (categoría a que pertenece y porcentaje de precisión) en consola.
