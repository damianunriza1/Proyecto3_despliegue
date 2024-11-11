# Proyecto de Despliegue de Soluciones Analíticas - Estimación de Precios de la Tierra en Colombia


## Integrantes
Roberto Gonzalez Bustamante
Richard Anderson Suan Yara
Jaime Unriza

## Descripción del Proyecto
En Colombia, la falta de información precisa sobre los precios de la tierra rural limita el desarrollo rural y la productividad agropecuaria. Este proyecto busca resolver esta brecha de información, proporcionando una herramienta de estimación de precios de terrenos rurales a partir de datos biofísicos y socioeconómicos oficiales. Mediante el uso de un modelo de machine learning supervisado, el proyecto tiene como objetivo proporcionar una valoración más precisa de los terrenos rurales, optimizando la toma de decisiones en el mercado de tierras y en la planificación territorial.

## Contexto y Problemas Abordados
- **Productividad Agrícola Baja:**La ausencia de datos confiables sobre precios limita decisiones informadas de propietarios y entidades financieras.
- **Especulación de Precios:** Los precios inflados afectan a pequeños productores y distorsionan el mercado.
Informalidad y Conflictos Territoriales: La informalidad en la propiedad de tierras agrava los conflictos y limita el acceso a programas de desarrollo.
- **Planificación Territorial y Sostenibilidad:** La falta de datos confiables afecta la creación de políticas de uso del suelo y la conservación ambiental.

## Descripción del Conjunto de Datos

El conjunto de datos cuenta con 175,526 registros y fue recolectado entre 2020 y 2024 a partir de muestreos de campo y bases de datos oficiales de entidades como IGAC, IDEAM y UPRA. Los datos incluyen:

127 variables originales: que incluyen 4 de ubicación, 33 edáficas, 47 ecosistémicas, 27 hídricas y 20 climáticas.
Tipos de datos: 14 variables float64, 25 int64 y 90 object (categóricas).


# Ejecución del Proyecto

Para ejecutar este proyecto, sigue los siguientes pasos:

## 1. Instalar un entorno virtual

Primero, necesitas tener un entorno virtual para el proyecto. Si no tienes uno, puedes crear uno utilizando el siguiente comando:

```bash
python3 -m venv ./env-proyecto
```

## 2. Activar el entorno virtual

Una vez creado el entorno virtual, actívalo ejecutando el siguiente comando:

- En **Linux/Mac**:

    ```bash
    source ./env-proyecto/bin/activate
    ```

- En **Windows**:

    ```bash
    .\env-proyecto\Scripts\activate
    ```

## 3. Instalar las dependencias

Con el entorno virtual activado, instala las dependencias necesarias para el proyecto utilizando `pip`:

```bash
pip install tox
```
luego ejecutar el comando 
```bash
tox -e dev
```
esto puede tardar bastante ya que las librerias de pandas y sklearn son bastante pesadas

¡Listo! Ahora puedes ejecutar y trabajar en el proyecto dentro de tu entorno virtual.

---

