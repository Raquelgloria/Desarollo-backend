import tkinter as tk

# Función que se ejecuta al presionar el botón
def traducir():
    frase = entrada.get()
    etiqueta_resultado.config(text=frase)

# Crear la ventana principal
ventana = tk.Tk()
ventana.title("Traductor Simple")

# Crear el campo de entrada
etiqueta = tk.Label(ventana, text="Ingrese una frase o palabra:")
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
