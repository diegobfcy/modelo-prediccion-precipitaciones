# modelo-prediccion-precipitaciones
1. config.py para mejorar el modelo.
2. data_preprocessing.py para sacar los datos del .nc (https://drive.google.com/file/d/1sLE3qZXpoPcWc3XauA_3QC3JmaBqhlW_/view?usp=sharing).
3. model_definition.py aun no se bien para que sirve pero ahi esta.
4. train_model.py el entrenamiento fue realizado usando el env tf_cuda_env IMPORTANTE: instalar primero las librerias cuda y cudnn para poder usar la gpu para el entrenamiento.
5. evaluate.py nos muestra comparaciones con los test para ver que tan preciso es el modelo.
6. predict.py predice el mes siguiente del que se sacaron los datos (index 530 - Marzo 2025)
## Mejoras futuras
1. Hacer otro modelo que me permita predecir el mes que sea (COMPLETADO - predict-mes.py).
2. Evaluar que tan a futuro se puede predecir con ese modelo sin perder precision.