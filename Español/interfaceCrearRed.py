import tkinter as tk
from tkinter import messagebox
import subprocess
import platform
import os


def executar():
    anioInicial_param = anioInicial.get()
    anioFinal_param = anioFinal.get()
    ponderadorAfinidad_param = ponderadorAfinidad.get()
    ponderadorCoI_param = ponderadorCoI.get()
    evaluadoresXProyecto_param = evaluadoresXProyecto.get()
    proyectoXEvaluadores_param = proyectoXEvaluadores.get()

    sistemaOperativo = platform.system()

    if sistemaOperativo == "Windows":
        python = os.path.join('python')

    elif sistemaOperativo == "Linux":
        python = os.path.join('python')

    else:
        python_venv = None
        messagebox.showerror("Erro", f"Sistema operativo '{sistemaOperativo}' no sportado.")


    comando = [
        python, 'crearRed.py',
        anioInicial_param,
        anioFinal_param,
        ponderadorAfinidad_param,
        ponderadorCoI_param,
        evaluadoresXProyecto_param,
        proyectoXEvaluadores_param
    ]



    subprocess.run(comando)


ventana = tk.Tk()
ventana.title("Criar REDES")


anioInicial_label = tk.Label(ventana, text="Insira el año inicial:")
anioInicial_label.grid(row=0, column=0)
anioInicial = tk.Entry(ventana)
anioInicial.grid(row=0, column=1)

anioFinal_label = tk.Label(ventana, text="Insira el año final:")
anioFinal_label.grid(row=1, column=0)
anioFinal = tk.Entry(ventana)
anioFinal.grid(row=1, column=1)

vacio_label = tk.Label(ventana, text="    ")
vacio_label.grid(row=2, column=0)

ponderadorAfinidad_label = tk.Label(ventana, text="Valor del ponderador X de afinidad: ")
ponderadorAfinidad_label.grid(row=3, column=0)
ponderadorAfinidad = tk.Entry(ventana)
ponderadorAfinidad.insert(0, "1")
ponderadorAfinidad.grid(row=3, column=1)

ponderadorCoI_label = tk.Label(ventana, text="Valor del ponderador Y de CoI: ")
ponderadorCoI_label.grid(row=3, column=3)
ponderadorCoI = tk.Entry(ventana)
ponderadorCoI.insert(0, "1")
ponderadorCoI.grid(row=3, column=4)

vacio2_label = tk.Label(ventana, text="    ")
vacio2_label.grid(row=4, column=0)

proyectoXEvaluadores_label = tk.Label(ventana, text="Evaluadores por proyecto: ")
proyectoXEvaluadores_label.grid(row=5, column=0)
proyectoXEvaluadores = tk.Entry(ventana)
proyectoXEvaluadores.insert(0, "1")
proyectoXEvaluadores.grid(row=5, column=1)

evaluadoresXProyecto_label = tk.Label(ventana, text="Máximo proyectos por evaluador: ")
evaluadoresXProyecto_label.grid(row=5, column=3)
evaluadoresXProyecto = tk.Entry(ventana)
evaluadoresXProyecto.insert(0, "1")
evaluadoresXProyecto.grid(row=5, column=4)

botonCrear = tk.Button(ventana, text="Crear", command=executar)
botonCrear.grid(row=6, column=0, columnspan=2)

ventana.mainloop()
