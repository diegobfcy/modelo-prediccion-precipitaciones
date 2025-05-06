import xarray as xr
import matplotlib.pyplot as plt

# Cargar el archivo NetCDF
ds = xr.open_dataset("C:/Users/HP/Documents/InteligenciaArtificial/modelo-prediccion-prueba/data/data.nc", decode_times=False)


# Mostrar las variables disponibles
print(ds)

# Extraer un "frame" de precipitación en el tiempo 0
prec_0 = ds['Prec'].isel(T=0)

# Graficar
plt.figure(figsize=(8, 6))
prec_0.plot(cmap='Blues')
plt.title("Precipitación en el tiempo T=0")
plt.show()
