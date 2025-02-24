from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from fastapi.responses import StreamingResponse
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from typing import List, Dict, Union
from geopy.geocoders import Nominatim
from scipy.stats import gaussian_kde

app = FastAPI()

# Base de datos local de coordenadas de ciudades paraguayas
CITIES_COORDINATES = {
    "Asunción": (-25.2637, -57.5759),
    "Encarnación": (-27.3306, -55.8667),
    "Caacupé": (-24.7333, -56.4167),
    "Ciudad del Este": (-25.5167, -54.6167),
    "Villarrica": (-26.4492, -56.3875),
    "Pedro Juan Caballero": (-22.5473, -55.7333),
    "San Lorenzo": (-25.34, -57.53),
    "Concepción": (-23.405, -57.4325),
    "Coronel Oviedo": (-25.4479, -56.4406),
    "Paraguarí": (-25.62, -57.15)
}

# Definir la estructura del JSON
class GeoDataInput(BaseModel):
    data: List[Dict[str, Union[str, float, int]]]  # 'ciudad' es str, 'count' es float o int

@app.post("/heatmap/")
async def generate_geo_heatmap(input_data: GeoDataInput):
    try:
        # Convertir JSON a DataFrame
        df = pd.DataFrame(input_data.data)

        if "ciudad" not in df or "count" not in df:
            return {"error": "El JSON debe incluir 'ciudad' y 'count'."}

        # Convertir nombres de ciudades a coordenadas
        geolocator = Nominatim(user_agent="paraguay_heatmap")

        latitudes, longitudes, counts = [], [], []

        for index, row in df.iterrows():
            city_name = row["ciudad"]
            count = row["count"]

            if city_name in CITIES_COORDINATES:
                lat, lon = CITIES_COORDINATES[city_name]
            else:
                location = geolocator.geocode(f"{city_name}, Paraguay")
                if location:
                    lat, lon = location.latitude, location.longitude
                    CITIES_COORDINATES[city_name] = (lat, lon)  # Guardar en caché
                else:
                    continue  # Si no se encuentra la ciudad, la omitimos

            latitudes.append(lat)
            longitudes.append(lon)
            counts.append(count)

        if not latitudes:
            return {"error": "No se encontraron coordenadas para las ciudades ingresadas."}

        # Crear la figura con una proyección en coordenadas geográficas
        fig, ax = plt.subplots(figsize=(12, 8), subplot_kw={"projection": ccrs.PlateCarree()})

        # Centrar el mapa en Paraguay y ajustar la vista
        ax.set_extent([-62, -54, -28, -19], crs=ccrs.PlateCarree())

        # Agregar características geográficas con fondo oscuro
        ax.set_facecolor("black")
        ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor="white")
        ax.add_feature(cfeature.COASTLINE, edgecolor="white")
        ax.add_feature(cfeature.LAND, facecolor="dimgray")
        ax.add_feature(cfeature.LAKES, edgecolor="blue", facecolor="none")
        ax.add_feature(cfeature.RIVERS, edgecolor="blue")

        # Crear un grid para interpolación suave del calor
        x = np.linspace(min(longitudes)-1, max(longitudes)+1, 100)
        y = np.linspace(min(latitudes)-1, max(latitudes)+1, 100)
        X, Y = np.meshgrid(x, y)

        # Aplicar kernel density estimation (KDE) para suavizar
        kde = gaussian_kde([longitudes, latitudes], weights=counts)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        # Dibujar el mapa de calor interpolado
        img = ax.imshow(Z, extent=[x.min(), x.max(), y.min(), y.max()], origin='lower',
                        cmap="turbo", alpha=0.75, aspect='auto', transform=ccrs.PlateCarree())

        # Agregar una barra de colores
        cbar = plt.colorbar(img, ax=ax, orientation="horizontal", pad=0.05)
        cbar.set_label("Intensidad de Acreditaciones")

        # Guardar la imagen en memoria
        img_io = io.BytesIO()
        plt.savefig(img_io, format="png", bbox_inches="tight", dpi=300)
        plt.close(fig)
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")

    except Exception as e:
        return {"error": str(e)}
