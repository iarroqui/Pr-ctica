import pandas as pd
import numpy as np
from time import time

def cargar_datos_csv(ruta_archivo):
    """
    Carga el archivo CSV con bloques separados por espacios.
    """
    return pd.read_csv(ruta_archivo, sep='\s+')

def dividir_bloques(df, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz):
    """
    Divide bloques grandes en sub-bloques más pequeños si el tamaño nuevo es múltiplo del original.
    
    Args:
        df (pd.DataFrame): DataFrame con bloques originales.
        dx, dy, dz (float): Dimensiones originales de los bloques.
        nuevo_dx, nuevo_dy, nuevo_dz (float): Dimensiones deseadas para los nuevos bloques.
    
    Returns:
        pd.DataFrame: DataFrame con los sub-bloques generados.
    """
    if dx % nuevo_dx != 0 or dy % nuevo_dy != 0 or dz % nuevo_dz != 0:
        raise ValueError("El nuevo tamaño de bloque no es múltiplo del tamaño original.")
    
    subdivisiones_x = int(dx / nuevo_dx)
    subdivisiones_y = int(dy / nuevo_dy)
    subdivisiones_z = int(dz / nuevo_dz)
    
    nuevos_bloques = []
    for index, row in df.iterrows():
        # Coordenadas del centro del bloque original
        x_centro, y_centro, z_centro = row["x"], row["y"], row["z"]
        for i in range(subdivisiones_x):
            for j in range(subdivisiones_y):
                for k in range(subdivisiones_z):
                    # Calcular el nuevo centro
                    nuevo_x = x_centro - dx / 2 + nuevo_dx / 2 + i * nuevo_dx
                    nuevo_y = y_centro - dy / 2 + nuevo_dy / 2 + j * nuevo_dy
                    nuevo_z = z_centro - dz / 2 + nuevo_dz / 2 + k * nuevo_dz
                    
                    # Copiar las propiedades del bloque original
                    nuevo_bloque = row.to_dict()
                    nuevo_bloque["x"], nuevo_bloque["y"], nuevo_bloque["z"] = nuevo_x, nuevo_y, nuevo_z
                    nuevos_bloques.append(nuevo_bloque)
    
    return pd.DataFrame(nuevos_bloques)

def calcular_intersecciones_numpy(centros_originales, dimensiones_originales, centro_nuevo, dimensiones_nuevo):
    """
    Calcula la proporción de intersección entre múltiples bloques originales y un bloque nuevo utilizando NumPy.
    """
    min_original = centros_originales - dimensiones_originales / 2
    max_original = centros_originales + dimensiones_originales / 2
    
    min_nuevo = centro_nuevo - dimensiones_nuevo / 2
    max_nuevo = centro_nuevo + dimensiones_nuevo / 2
    
    # Calcular intersecciones por cada eje
    interseccion_x = np.maximum(0, np.minimum(max_original[:, 0], max_nuevo[0]) - np.maximum(min_original[:, 0], min_nuevo[0]))
    interseccion_y = np.maximum(0, np.minimum(max_original[:, 1], max_nuevo[1]) - np.maximum(min_original[:, 1], min_nuevo[1]))
    interseccion_z = np.maximum(0, np.minimum(max_original[:, 2], max_nuevo[2]) - np.maximum(min_original[:, 2], min_nuevo[2]))
    
    # Volumen de intersección
    volumen_interseccion = interseccion_x * interseccion_y * interseccion_z
    volumen_original = np.prod(dimensiones_originales, axis=1)
    
    return volumen_interseccion / volumen_original

def rebloquear_bloques_ponderado(df, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz):
    """
    Realiza el rebloqueo calculando intersecciones y ponderaciones cuando los bloques no son múltiplos exactos.
    """
    t1=time()
    columnas_propiedades = df.columns.difference(["x", "y", "z"])
    centros_originales = df[["x", "y", "z"]].values
    dimensiones_originales = np.array([dx, dy, dz])

    x_coords = np.arange(df["x"].min() - dx, df["x"].max() + dx, nuevo_dx)
    y_coords = np.arange(df["y"].min() - dy, df["y"].max() + dy, nuevo_dy)
    z_coords = np.arange(df["z"].min() - dz, df["z"].max() + dz, nuevo_dz)

    x_grid, y_grid, z_grid = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')
    nuevos_centros = np.column_stack((x_grid.ravel() + nuevo_dx / 2,
                                      y_grid.ravel() + nuevo_dy / 2,
                                      z_grid.ravel() + nuevo_dz / 2))

    dimensiones_nuevo = np.array([nuevo_dx, nuevo_dy, nuevo_dz])
    proporciones = calcular_intersecciones_numpy(centros_originales, dimensiones_originales, nuevos_centros, dimensiones_nuevo)

    indices_relevantes = np.any(proporciones > 0, axis=1)
    nuevos_centros_relevantes = nuevos_centros[indices_relevantes]
    proporciones_relevantes = proporciones[indices_relevantes]

    nuevos_bloques = []
    for i, centro in enumerate(nuevos_centros_relevantes):
        proporciones_actuales = proporciones_relevantes[i]
        indices_activos = np.where(proporciones_actuales > 0)[0]

        suma_pesos = np.sum(proporciones_actuales[indices_activos])
        bloque_actual = {}

        for col in columnas_propiedades:
            valores = df.iloc[indices_activos][col].values
            bloque_actual[col] = np.sum(valores * proporciones_actuales[indices_activos]) / suma_pesos

        bloque_actual.update({"x": centro[0], "y": centro[1], "z": centro[2]})
        nuevos_bloques.append(bloque_actual)
    t2=time()-t1
    print("tiempo intersecciones rebloquear bloques ponderado =", t2)
    return pd.DataFrame(nuevos_bloques)

def rebloquear_bloques(df, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz):
    """
    Decide entre dividir bloques directamente o realizar rebloqueo ponderado.
    """
    if dx % nuevo_dx == 0 and dy % nuevo_dy == 0 and dz % nuevo_dz == 0:
        print("El nuevo tamaño es múltiplo del original. Dividiendo bloques directamente...")
        return dividir_bloques(df, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz)
    else:
        print("El nuevo tamaño no es múltiplo del original. Realizando rebloqueo ponderado...")
        return rebloquear_bloques_ponderado(df, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz)

def guardar_csv(df, ruta_salida):
    """
    Guarda el DataFrame resultante a un archivo CSV.
    """
    df.to_csv(ruta_salida, sep=' ', index=False)

def main():
    t0 = time()
    ruta_entrada = "mb_general1.csv"
    ruta_salida = "modelo_rebloqueado1.csv"

    # Tamaño actual y nuevo de los bloques
    dx, dy, dz = 10.0, 10.0, 10.0  # Tamaño original
    nuevo_dx, nuevo_dy, nuevo_dz = 2.0, 2.0, 2.0  # Tamaño deseado

    # Cargar datos
    df_bloques = cargar_datos_csv(ruta_entrada)

    # Elegir el método adecuado y realizar el rebloqueo
    df_rebloqueado = rebloquear_bloques(df_bloques, dx, dy, dz, nuevo_dx, nuevo_dy, nuevo_dz)

    # Guardar resultados
    guardar_csv(df_rebloqueado, ruta_salida)
    t = time() - t0
    print("Proceso completado. Modelo rebloqueado guardado en:", ruta_salida)
    print("tiempo Simulador = " + str(round(t, 2)) + " s")
if __name__ == "__main__":
    main()
