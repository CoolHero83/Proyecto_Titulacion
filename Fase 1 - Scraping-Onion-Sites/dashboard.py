"""
Dashboard interactivo para visualizar datos preprocesados de foros .onion
Creado con Streamlit para análisis exploratorio y presentaciones.

Uso:
    streamlit run dashboard.py

Requiere:
    pip install streamlit pandas plotly
"""

import streamlit as st
import pandas as pd
import json
from datetime import datetime
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Configuración de la página
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard - Datos .onion",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# Funciones auxiliares
# ---------------------------------------------------------------------------

@st.cache_data
def cargar_datos(ruta: str) -> pd.DataFrame:
    """Carga el CSV preprocesado y lo cachea para mejorar rendimiento."""
    df = pd.read_csv(ruta)

    # Convertir columnas de fecha si existen
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    if 'fecha_hora' in df.columns:
        df['fecha_hora'] = pd.to_datetime(df['fecha_hora'], errors='coerce')

    return df


# ---------------------------------------------------------------------------
# Barra lateral
# ---------------------------------------------------------------------------

st.sidebar.title("🌐 Dashboard .onion")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📁 Carga de datos")

# Selector de archivo (por defecto busca el archivo preprocesado en Datos/)
RUTA_DEFAULT = "../Datos/forum_records_clean.csv"
ruta_archivo = st.sidebar.text_input(
    "Ruta del CSV preprocesado:",
    value=RUTA_DEFAULT
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Filtros")

# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

try:
    df = cargar_datos(ruta_archivo)
    st.sidebar.success(f"✅ Datos cargados: {len(df)} registros")
    datos_cargados = True
except Exception as e:
    st.sidebar.error(f"❌ Error al cargar datos: {e}")
    st.sidebar.info("💡 Asegúrate de ejecutar primero el Preprocesador.py")
    datos_cargados = False

if not datos_cargados:
    st.title("🌐 Dashboard de Datos .onion")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col2:
        st.warning("### 📂 No se encontraron datos")
        st.markdown("""
        Para usar este dashboard, primero debes:

        1. **Ejecutar el scraper** para obtener datos crudos
        2. **Ejecutar el Preprocesador** para limpiar los datos:
           ```bash
           python Preprocesador.py --input ../Datos/forum_records.csv --output ../Datos/forum_records_clean.csv
           ```
        3. **Cargar el archivo** en el sidebar de este dashboard
        """)

    st.stop()

# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------

# Filtro por rango de fechas
if 'fecha' in df.columns and not df['fecha'].isna().all():
    fecha_min = df['fecha'].min()
    fecha_max = df['fecha'].max()

    if pd.notna(fecha_min) and pd.notna(fecha_max):
        rango_fechas = st.sidebar.date_input(
            "Rango de fechas:",
            value=(fecha_min.date(), fecha_max.date()),
            min_value=fecha_min.date(),
            max_value=fecha_max.date()
        )

        if len(rango_fechas) == 2:
            fecha_inicio, fecha_fin = rango_fechas
            df = df[(df['fecha'].dt.date >= fecha_inicio) &
                    (df['fecha'].dt.date <= fecha_fin)]

# Filtro por longitud mínima del cuerpo
if 'longitud_cuerpo' in df.columns:
    min_long = int(df['longitud_cuerpo'].min())
    max_long = int(df['longitud_cuerpo'].max())
    rango_longitud = st.sidebar.slider(
        "Longitud mínima del cuerpo:",
        min_value=min_long,
        max_value=max_long,
        value=(min_long, min(max_long, 500))
    )
    df = df[(df['longitud_cuerpo'] >= rango_longitud[0]) &
            (df['longitud_cuerpo'] <= rango_longitud[1])]

# ---------------------------------------------------------------------------
# Cabecera
# ---------------------------------------------------------------------------

st.title("🌐 Dashboard de Datos de Foros .onion")
st.markdown("---")

# ---------------------------------------------------------------------------
# KPIs - Métricas principales
# ---------------------------------------------------------------------------

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric(
        label="📝 Total Posts",
        value=f"{len(df):,}",
        delta=None
    )

with col2:
    if 'usuario' in df.columns:
        usuarios_unicos = df['usuario'].nunique()
        st.metric(
            label="👤 Usuarios Únicos",
            value=f"{usuarios_unicos:,}"
        )

with col3:
    if 'longitud_cuerpo' in df.columns:
        long_promedio = int(df['longitud_cuerpo'].mean())
        st.metric(
            label="📏 Longitud Promedio",
            value=f"{long_promedio:,} chars"
        )

with col4:
    if 'fecha' in df.columns and not df['fecha'].isna().all():
        dias_activos = df['fecha'].dt.date.nunique()
        st.metric(
            label="📅 Días Activos",
            value=f"{dias_activos:,}"
        )

with col5:
    if 'id_hilo' in df.columns:
        hilos_unicos = df['id_hilo'].nunique()
        st.metric(
            label="🧵 Hilos Únicos",
            value=f"{hilos_unicos:,}"
        )

st.markdown("---")

# ---------------------------------------------------------------------------
# Layout en pestañas
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Resumen General",
    "👥 Actividad por Autor",
    "📅 Tendencia Temporal",
    "📋 Datos Crudos"
])

# ============================================================================
# PESTAÑA 1: Resumen General
# ============================================================================

with tab1:
    st.subheader("📊 Resumen General de los Datos")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📈 Estadísticas Descriptivas")

        stats_data = []

        if 'longitud_cuerpo' in df.columns:
            stats_data.append({
                "Métrica": "Longitud Cuerpo",
                "Media": f"{df['longitud_cuerpo'].mean():.0f}",
                "Mediana": f"{df['longitud_cuerpo'].median():.0f}",
                "Mínimo": f"{df['longitud_cuerpo'].min():.0f}",
                "Máximo": f"{df['longitud_cuerpo'].max():.0f}",
                "Desv. Est.": f"{df['longitud_cuerpo'].std():.0f}"
            })

        if 'longitud_titulo' in df.columns:
            stats_data.append({
                "Métrica": "Longitud Título",
                "Media": f"{df['longitud_titulo'].mean():.0f}",
                "Mediana": f"{df['longitud_titulo'].median():.0f}",
                "Mínimo": f"{df['longitud_titulo'].min():.0f}",
                "Máximo": f"{df['longitud_titulo'].max():.0f}",
                "Desv. Est.": f"{df['longitud_titulo'].std():.0f}"
            })

        if stats_data:
            st.dataframe(
                pd.DataFrame(stats_data).set_index("Métrica"),
                use_container_width=True
            )

    with col_right:
        st.markdown("### 📊 Distribución de Longitud del Cuerpo")

        if 'longitud_cuerpo' in df.columns:
            fig = px.histogram(
                df,
                x='longitud_cuerpo',
                nbins=30,
                title="Distribución de Longitud de Posts",
                labels={'longitud_cuerpo': 'Longitud (caracteres)'},
                color_discrete_sequence=['#1f77b4']
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Distribución horaria
    st.markdown("### 🕐 Distribución por Hora del Día")

    if 'hora' in df.columns:
        conteo_horas = df['hora'].value_counts().sort_index().reset_index()
        conteo_horas.columns = ['hora', 'cantidad']

        fig = px.bar(
            conteo_horas,
            x='hora',
            y='cantidad',
            title="Posts por Hora del Día",
            labels={'hora': 'Hora', 'cantidad': 'Cantidad de Posts'},
            color='cantidad',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PESTAÑA 2: Actividad por Autor
# ============================================================================

with tab2:
    st.subheader("👥 Análisis de Actividad por Autor")

    if 'usuario' in df.columns:
        # Top autores
        top_n = st.slider("Mostrar Top N autores:", min_value=5, max_value=30, value=10)

        top_autores = df['usuario'].value_counts().head(top_n).reset_index()
        top_autores.columns = ['usuario', 'cantidad']

        col_left, col_right = st.columns([2, 1])

        with col_left:
            fig = px.bar(
                top_autores,
                x='cantidad',
                y='usuario',
                orientation='h',
                title=f"Top {top_n} Autores más Activos",
                labels={'cantidad': 'Cantidad de Posts', 'usuario': 'Autor'},
                color='cantidad',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### 📊 Estadísticas de Autores")
            st.metric("Total Autores", df['usuario'].nunique())
            st.metric("Posts por Autor (promedio)", f"{len(df) / df['usuario'].nunique():.1f}")

            if 'longitud_cuerpo' in df.columns:
                # Autor con más texto
                autor_mas_texto = df.groupby('usuario')['longitud_cuerpo'].sum().idxmax()
                st.metric("Autor con más texto", autor_mas_texto)

# ============================================================================
# PESTAÑA 3: Tendencia Temporal
# ============================================================================

with tab3:
    st.subheader("📅 Análisis de Tendencia Temporal")

    if 'fecha' in df.columns and not df['fecha'].isna().all():

        col_left, col_right = st.columns(2)

        with col_left:
            # Posts por día
            posts_por_dia = df.groupby(df['fecha'].dt.date).size().reset_index()
            posts_por_dia.columns = ['fecha', 'cantidad']
            posts_por_dia = posts_por_dia.sort_values('fecha')

            fig = px.line(
                posts_por_dia,
                x='fecha',
                y='cantidad',
                title="Posts por Día",
                labels={'fecha': 'Fecha', 'cantidad': 'Cantidad de Posts'},
                markers=True
            )
            fig.update_traces(line_color='#2ecc71')
            st.plotly_chart(fig, use_container_width=True)

        with col_right:
            # Posts por mes
            if 'año' in df.columns and 'mes' in df.columns:
                df['periodo'] = df['fecha'].dt.to_period('M').astype(str)
                posts_por_mes = df.groupby('periodo').size().reset_index()
                posts_por_mes.columns = ['periodo', 'cantidad']

                fig = px.bar(
                    posts_por_mes,
                    x='periodo',
                    y='cantidad',
                    title="Posts por Mes",
                    labels={'periodo': 'Mes', 'cantidad': 'Cantidad de Posts'},
                    color='cantidad',
                    color_continuous_scale='Reds'
                )
                st.plotly_chart(fig, use_container_width=True)

        # Mapa de calor: días de la semana vs horas
        st.markdown("### 🔥 Mapa de Calor: Día de Semana vs Hora")

        if 'hora' in df.columns:
            df['dia_semana'] = df['fecha'].dt.day_name()
            dias_orden = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            heatmap_data = df.groupby(['dia_semana', 'hora']).size().unstack(fill_value=0)
            heatmap_data = heatmap_data.reindex(dias_orden, axis=0)

            fig = px.imshow(
                heatmap_data,
                title="Actividad por Día de Semana y Hora",
                labels={'x': 'Hora del Día', 'y': 'Día de Semana', 'color': 'Posts'},
                aspect="auto",
                color_continuous_scale='YlOrRd'
            )
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PESTAÑA 4: Datos Crudos
# ============================================================================

with tab4:
    st.subheader("📋 Vista de Datos Crudos")

    # Selector de columnas a mostrar
    columnas_disponibles = df.columns.tolist()
    columnas_seleccionadas = st.multiselect(
        "Seleccionar columnas a mostrar:",
        columnas_disponibles,
        default=[col for col in ['id_mensaje', 'usuario', 'fecha_hora',
                                   'titulo_limpio', 'cuerpo_limpio', 'longitud_cuerpo']
                 if col in columnas_disponibles]
    )

    if columnas_seleccionadas:
        # Paginación
        registros_por_pagina = st.selectbox(
            "Registros por página:",
            options=[10, 25, 50, 100],
            index=0
        )

        inicio = st.number_input(
            "Mostrar desde el registro:",
            min_value=0,
            max_value=len(df),
            value=0,
            step=registros_por_pagina
        )

        fin = min(inicio + registros_por_pagina, len(df))
        df_mostrar = df[columnas_seleccionadas].iloc[inicio:fin]

        st.dataframe(df_mostrar, use_container_width=True)
        st.caption(f"Mostrando registros {inicio}-{fin} de {len(df)} totales")

        # Botón de descarga
        st.markdown("### 💾 Descargar datos filtrados")

        formato_descarga = st.radio("Formato:", ["CSV", "JSON"], horizontal=True)

        if formato_descarga == "CSV":
            datos_descarga = df.to_csv(index=False).encode('utf-8')
            tipo_archivo = "text/csv"
            extension = "csv"
        else:
            datos_descarga = df.to_json(orient='records', force_ascii=False).encode('utf-8')
            tipo_archivo = "application/json"
            extension = "json"

        st.download_button(
            label=f"⬇️ Descargar como {formato_descarga}",
            data=datos_descarga,
            file_name=f"datos_onion_filtrados.{extension}",
            mime=tipo_archivo
        )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
        Dashboard creado con Streamlit · Datos de foros .onion ·
        Proyecto de Titulación TI
    </div>
    """,
    unsafe_allow_html=True
)