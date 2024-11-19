import tkinter as tk
from translate import translate_text  

# Función que se ejecuta al presionar el botón
def traducir():
    source_text = entrada.get()
    etiqueta_resultado.config(text=source_text)

    try:
        # Llamar a la función de traducción
        translated_text = translate_text(source_text, source_lang="en_XX", target_lang="es_XX")
        etiqueta_resultado.config(text=translated_text)  # Mostrar el texto traducido
        
        
    except Exception as e:
        etiqueta_resultado.config(text=f"Error: {str(e)}")

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Traductor Simple")

# Crear el campo de entrada
etiqueta = tk.Label(ventana, text="Ingrese una frase o palabra (en inglés):")
etiqueta.pack(pady=5)

entrada = tk.Entry(ventana, width=40)
entrada.pack(pady=5)

# Crear el botón
boton = tk.Button(ventana, text="Traducir", command=traducir)
boton.pack(pady=10)

# Etiqueta para mostrar el resultado
etiqueta_resultado = tk.Label(ventana, text="", font=("Arial", 12))
etiqueta_resultado.pack(pady=10)

# Iniciar el bucle de la aplicación
ventana.mainloop()
