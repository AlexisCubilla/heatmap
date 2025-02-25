from fastapi import FastAPI, Response, Depends
import folium
import pandas as pd
import numpy as np
from folium.plugins import HeatMap
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut
import asyncpg
import os
import logging
from dotenv import load_dotenv
from functools import lru_cache
from branca.colormap import LinearColormap

# Configurar logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Cargar variables de entorno desde .env
load_dotenv()

app = FastAPI()

# Configuración de la conexión a PostgreSQL
DB_CONFIG = {
    "host": os.getenv("DB_HOST"),
    "database": os.getenv("DB_NAME"),
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
    "port": os.getenv("DB_PORT"),
}

# Definir colores fijos como constantes globales
HEATMAP_COLORS = {
    'lowest': '#03254C',    # Azul oscuro (más bajo)
    'low': '#1C6DD0',      # Azul claro (intermedio bajo)
    'medium': '#4BA36F',    # Verde (intermedio)
    'high': '#FFD93D',     # Amarillo (intermedio alto)
    'higher': '#FF8400',    # Naranja (intermedio alto)
    'highest': '#FF0000'    # Rojo (más alto)
}


# Inicializar pool de conexiones a PostgreSQL
async def get_db_pool():
    return await asyncpg.create_pool(**DB_CONFIG)

# Inicializar geolocalizador
geolocator = Nominatim(user_agent="geoapi")

# Caché para geolocalización
@lru_cache(maxsize=500)
def obtener_coordenadas(ciudad: str, pais: str = "Paraguay"):
    """Obtiene la latitud y longitud de una ciudad usando Nominatim con caché."""
    try:
        location = geolocator.geocode(f"{ciudad}, {pais}", timeout=10)
        if location:
            return location.latitude, location.longitude
    except GeocoderTimedOut:
        logging.warning(f"Timeout al obtener coordenadas de {ciudad}")
        return None, None
    except Exception as e:
        logging.error(f"Error en geocodificación: {e}")
        return None, None
    return None, None


# Función para obtener coordenadas de Paraguay
def obtener_coordenadas_paraguay():
    """Obtiene las coordenadas y límites de Paraguay usando Nominatim."""
    try:
        geolocator = Nominatim(user_agent="geoapi")
        location = geolocator.geocode("Paraguay", featuretype="country")
        
        if location:
            # location.raw contiene toda la información del país
            bbox = location.raw['boundingbox']
            return {
                "center": [location.latitude, location.longitude],
                "bounds": {
                    "min_lat": float(bbox[0]),  # Sur
                    "max_lat": float(bbox[1]),  # Norte
                    "min_lon": float(bbox[2]),  # Oeste
                    "max_lon": float(bbox[3])   # Este
                },
                "zoom": {
                    "start": 6,
                    "min": 6,
                    "max": 8
                }
            }
    except Exception as e:
        logging.error(f"Error obteniendo coordenadas de Paraguay: {e}")
        # Valores por defecto en caso de error
        return {
            "center": [-23.4425, -58.4438],
            "bounds": {
                "min_lat": -27.6,
                "max_lat": -19.2,
                "min_lon": -62.0,
                "max_lon": -54.2
            },
            "zoom": {
                "start": 6,
                "min": 6,
                "max": 8
            }
        }
        
# Obtener configuración de Paraguay al iniciar la aplicación
PARAGUAY_CONFIG = obtener_coordenadas_paraguay()

print(PARAGUAY_CONFIG)

@app.post("/heatmap/")
async def generar_mapa(data: dict):
    """Genera un mapa de calor centrado en Paraguay con barra de intensidad."""
    df = pd.DataFrame(data["data"])

    # Obtener coordenadas dinámicamente con caché
    df["lat"], df["lon"] = zip(*df["ciudad"].apply(lambda x: obtener_coordenadas(x)))

    # Filtrar datos inválidos
    df = df.dropna(subset=["lat", "lon", "count"])
    if df.empty:
        return {"error": "No se encontraron coordenadas válidas para las ciudades proporcionadas."}

    # Convertir count a valores numéricos y eliminar NaN
    df["count"] = pd.to_numeric(df["count"], errors="coerce").fillna(0).astype(int)

    # Verificar que hay al menos un dato válido
    if df["count"].sum() == 0:
        return {"error": "Todos los valores de 'count' son cero o inválidos."}

    # Crear el mapa usando las coordenadas de Paraguay
    mapa = folium.Map(
        location=PARAGUAY_CONFIG["center"],
        zoom_start=PARAGUAY_CONFIG["zoom"]["start"],
        min_zoom=PARAGUAY_CONFIG["zoom"]["min"],
        max_bounds=True,
        max_bounds_viscosity=1.0,
        dragging=True,
        scrollWheelZoom=True,
        min_lat=PARAGUAY_CONFIG["bounds"]["min_lat"],
        max_lat=PARAGUAY_CONFIG["bounds"]["max_lat"],
        min_lon=PARAGUAY_CONFIG["bounds"]["min_lon"],
        max_lon=PARAGUAY_CONFIG["bounds"]["max_lon"]
    )

    # Asegurar que heat_data tenga valores correctos
    heat_data = [[float(row["lat"]), float(row["lon"]), int(row["count"])] for _, row in df.iterrows()]

    # Normalizar valores para vmin y vmax
    vmin = int(df["count"].min())
    vmax = int(df["count"].max())
    if vmin == vmax:
        vmax = vmin + 1  # Evitar errores si todos los valores son iguales

    # Usar los colores definidos globalmente
    colormap = LinearColormap(
        colors=list(HEATMAP_COLORS.values()),
        vmin=vmin,
        vmax=vmax,
        caption="Rango de Acreditaciones"
    ).to_step(n=len(HEATMAP_COLORS))

    # Crear gradiente usando los mismos colores y calculando los pasos
    step = 1.0 / (len(HEATMAP_COLORS) - 1)
    gradient_dict = {
        str(i * step): color for i, color in enumerate(HEATMAP_COLORS.values())
    }
    
    # Agregar HeatMap con barra de intensidad
    HeatMap(
        heat_data,
        min_opacity=0.3,  # Mayor visibilidad
        max_opacity=0.8,
        radius=25,  # Aumentar tamaño de puntos para visualizar mejor
        blur=12,
        gradient=gradient_dict
    ).add_to(mapa)

    # Agregar la leyenda de colores al mapa
    colormap.add_to(mapa)
    
    
    # Agregar marcadores con tooltip para mostrar la cantidad de acreditaciones
    for _, row in df.iterrows():
        acreditacion_text = "Acreditación" if row['count'] == 1 else "Acreditaciones"
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=5 + (row["count"] / vmax) * 10,  # Tamaño proporcional a la cantidad
            color="white",  # Color del borde (puede cambiarse a 'black' si se prefiere)
            fill=True,
            fill_color="black",  # Relleno invisible para no interferir con el heatmap
            fill_opacity=0.01,  # Casi transparente para que no sea invasivo
            weight=0,  # Sin borde
            tooltip=f"{row['ciudad']}: {row['count']} {acreditacion_text}"  # Texto al pasar el mouse
        ).add_to(mapa)
        # Convertir el mapa a HTML
    return Response(content=mapa._repr_html_(), media_type="text/html")

async def obtener_datos_bd(pool):
    """Consulta la base de datos PostgreSQL y obtiene los datos."""
    query = """
    SELECT  
        s.nombre AS ciudad, 
        COUNT(a.id) AS count
    FROM acreditaciones a
    JOIN ideas.ies_facultad_sede ifs ON ifs.id = a.fkufs 
    JOIN ideas.param_carrera_programa pcp ON pcp.id = a.fkcarrera_programa_id
    JOIN ideas.ies i ON i.id = ifs.fkies_id 
    JOIN ideas.facultad f ON f.id = ifs.fkfacultad_id 
    JOIN ideas.sede s ON s.id = ifs.fksede_id 
    JOIN crm.departamento d ON d.id = s.fkdepartment_id  
    JOIN ideas.resoluciones r ON r.id = a.fkresolucion_id
    WHERE a.fktipomecanismo_id = 1 AND r.sub_tipo_resoluciones = 1 AND a.fkestado_id = 1
    GROUP BY s.nombre
    ORDER BY COUNT(a.id) DESC;
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(query)
        return {"data": [{"ciudad": row["ciudad"], "count": row["count"]} for row in rows]}
    except Exception as e:
        logging.error(f"Error al obtener datos de la base de datos: {e}")
        return {"error": "Error al consultar la base de datos"}

@app.get("/heatmap-from-db/")
async def generar_mapa_desde_bd(pool=Depends(get_db_pool)):
    """Obtiene datos desde PostgreSQL y genera el mapa."""
    logging.info("Obteniendo datos de la base de datos...")
    data = await obtener_datos_bd(pool)
    
    print("datos:\n",data)
    
    if "error" in data:
        return data
    
    logging.info("Generando mapa de calor...")
    return await generar_mapa(data)
