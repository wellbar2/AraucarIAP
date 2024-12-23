import math
import tkinter
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import re
import requests
import json
import unidecode
import time

def buscaIdsAutor (primerNombre, ultimoApellido):
    buscaAutor = requests.get("https://api.openalex.org/authors?search=" + primerNombre + "%20" + ultimoApellido)
    resultados = json.loads(buscaAutor.text)
    autores = resultados["results"]

    idsOpenAlex = []
    for autor in autores:
        if re.match(unidecode.unidecode("^" + primerNombre + ".*" + ultimoApellido), unidecode.unidecode(autor["display_name"]), re.IGNORECASE):
            idsOpenAlex.append(autor)

        for item in treeElegirNombre.get_children():
            treeElegirNombre.delete(item)

    return idsOpenAlex


def armarProyecto():
    diretorioAtualAbsoluto = os.path.abspath("./")
    #diretorioAtualAbsoluto = os.path.abspath("../")
    direccion = diretorioAtualAbsoluto + "//proyectos//"
    archivoNombre = nombre.get()
    archivoApellido = apellido.get()

    if not archivoNombre.strip():
        archivoNombre = '1'
    if not archivoApellido.strip():
        archivoApellido = '1'

    archivoNombre = re.sub(r'[^a-zA-Z0-9 ]+', '', archivoNombre)
    archivoApellido = re.sub(r'[^a-zA-Z0-9 ]+', '', archivoApellido)



    tituloProyecto = proyecto.get().replace("\n", " ")
    tituloProyecto = re.sub(r'[^a-zA-Z0-9 ]+', '-', tituloProyecto)


    resumenProyecto = resumen.get(1.0,tkinter.END).replace("\n", "")
    archivo = open(direccion + tituloProyecto + ".txt","w", encoding="utf8")

    idsAutor = []
    ContaTrabajos = []

    nombre.delete(0, tkinter.END)
    apellido.delete(0, tkinter.END)
    proyecto.delete(0, tkinter.END)
    resumen.delete(1.0, tkinter.END)

    for item in treeElegirNombre.get_children():
        treeElegirNombre.delete(item)

    for item in treeGenerarCurriculo.get_children():
        idsAutor.append(treeGenerarCurriculo.item(item)["values"][0])
        ContaTrabajos.append(treeGenerarCurriculo.item(item)["values"][1])
        treeGenerarCurriculo.delete(item)

    curriculoVirtual = []
    cont = 0
    for id in idsAutor:
        paginas = math.ceil(int(ContaTrabajos[cont]) / 200)
        cont += 1

        for pagina in range(paginas):
            consulta = requests.get("https://api.openalex.org/works?filter=author.id:" + id + "&page=" + str(pagina+1) + "&per-page=200") #200 por pagina
            #https: // api.openalex.org /works?page=2&per-page=200    #para poner más paginas
            resultados = json.loads(consulta.text)
            trabajosPorId = resultados["results"]

            for trabajo in trabajosPorId:
                autoria = trabajo["authorships"]
                autores = []
                for autor in autoria:
                    autores.append(autor["author"])
                limpiaCaracteresTitulo = trabajo["title"]
                limpiaCaracteresTitulo = str(limpiaCaracteresTitulo).replace("\n", "").replace("||", "--")
                fecha = trabajo["publication_year"]
                tipo = trabajo["type"]
                linea = limpiaCaracteresTitulo + "||" + str(fecha) + "||" + str(tipo) + "||" + str(autores)
                #quitar duplicidad acá
                flag = True
                for contenido in curriculoVirtual:
                    if limpiaCaracteresTitulo in contenido:
                        print(limpiaCaracteresTitulo)
                        flag = False
                        break
                if flag:
                    curriculoVirtual.append(linea)

    archivo.writelines(str(idsAutor) + "\n")
    archivo.writelines(tituloProyecto + "\n")
    archivo.writelines(resumenProyecto + "\n")

    for linea in curriculoVirtual:
        archivo.writelines(linea + "\n")
    messagebox.showinfo("Proyecto listo", "Proyecto:  " + tituloProyecto + " - listo.")
    archivo.close()


def cargarTreeView():
    #messagebox.showinfo('Hola')
    listaAutores = buscaIdsAutor(nombre.get(), apellido.get())

    for item in treeElegirNombre.get_children():
        treeElegirNombre.delete(item)

    for autor in listaAutores:
        treeElegirNombre.insert("", tk.END, text=autor["display_name"], values=(str(autor["id"])[21:], autor["works_count"]))


def on_doubleClick_treeElegirNombre(event):
    linea = treeElegirNombre.focus()
    nombreAutor = treeElegirNombre.item(linea)["text"]
    idAutor = treeElegirNombre.item(linea)["values"][0]
    trabajosAutor = treeElegirNombre.item(linea)["values"][1]

    treeGenerarCurriculo.insert("", tk.END, text=nombreAutor, values=(idAutor,trabajosAutor))
    treeElegirNombre.delete(linea)


def on_rigthClick_treeElegirNombre(event):
    #diretorioAtualAbsoluto = os.path.abspath("./")
    diretorioAtualAbsoluto = os.path.abspath("../")
    direccion = diretorioAtualAbsoluto + "//evaluadores//"
    archivo  = open(direccion + "temp.txt","w", encoding="utf8")

    ContaTrabajos = []
    curriculoVirtual = []
    item = treeElegirNombre.focus()
    id = treeElegirNombre.item(item)["values"][0]
    ContaTrabajos.append(treeElegirNombre.item(item)["values"][1])

    consulta = requests.get("https://api.openalex.org/works?filter=author.id:" + id + "&per-page=20")  # 20 por pagina
    resultados = json.loads(consulta.text)
    trabajosPorId = resultados["results"]

    for trabajo in trabajosPorId:
        autoria = trabajo["authorships"]
        autores = []
        for autor in autoria:
            autores.append(autor["author"])
        limpiaCaracteresTitulo = trabajo["title"]
        limpiaCaracteresTitulo = str(limpiaCaracteresTitulo).replace("\n", "").replace("||", "--")
        fecha = trabajo["publication_year"]
        tipo = trabajo["type"]
        linea = limpiaCaracteresTitulo + "||" + str(fecha) + "||" + str(tipo) + "||" + str(autores)
        # quitar duplicidad acá
        flag = True
        for contenido in curriculoVirtual:
            if limpiaCaracteresTitulo in contenido:
                print(limpiaCaracteresTitulo)
                flag = False
                break
        if flag:
            curriculoVirtual.append(linea)

    archivo.writelines(str(id) + "\n")



    for linea in curriculoVirtual:
        archivo.writelines(linea + "\n")

    archivo.close()
    os.startfile(direccion + "temp.txt")
    time.sleep(1)
    os.remove(direccion + "temp.txt")





def on_doubleClick_treeGenerarCurriculo(event):
    linea = treeGenerarCurriculo.focus()
    nombreAutor = treeGenerarCurriculo.item(linea)["text"]
    idAutor = treeGenerarCurriculo.item(linea)["values"][0]
    trabajosAutor = treeGenerarCurriculo.item(linea)["values"][1]

    treeElegirNombre.insert("", tk.END, text=nombreAutor, values=(idAutor,trabajosAutor))
    treeGenerarCurriculo.delete(linea)

ventana = tk.Tk()
ventana.title("Busca autores openAlex")
#ventana.state('zoomed')

nombre_label = tk.Label(ventana, text="Nombre autor:")
nombre_label.grid(row=0, column=0)
nombre = tk.Entry(ventana)
nombre.grid(row=0, column=1)

apellido_label = tk.Label(ventana, text="Apellido autor:")
apellido_label.grid(row=1, column=0)
apellido = tk.Entry(ventana)
apellido.grid(row=1, column=1)

botonBusca = tk.Button(ventana, text="Buscar", command=cargarTreeView)
botonBusca.grid(row=2, column=1)


respuesta_label = tk.Label(ventana, text="Respuesta a la consulta:")
respuesta_label.grid(row=3, column=0)

treeElegirNombre = ttk.Treeview(ventana)
vsb = ttk.Scrollbar(ventana, orient="vertical", command=treeElegirNombre.yview, )
treeElegirNombre.configure(yscrollcommand=vsb.set)
vsb.grid(sticky="W", row=4, column=2, rowspan=2)

treeElegirNombre["columns"] = ("id", "trabajos")
treeElegirNombre.column("#0", width=250)
treeElegirNombre.column("id", width=100)
treeElegirNombre.column("trabajos", width=100)
treeElegirNombre.heading("#0", text="Nombre")
treeElegirNombre.heading("id", text="ID")
treeElegirNombre.heading("trabajos", text="Trabajos")
treeElegirNombre.bind("<Double-1>", on_doubleClick_treeElegirNombre)
treeElegirNombre.bind("<Button-3>", on_rigthClick_treeElegirNombre) #consultar curriculo del registro para validar
treeElegirNombre.grid(row=4, column=0 , columnspan=2)


elecciones_label = tk.Label(ventana, text="Autores a agregar al proyecto:")
elecciones_label.grid(row=3, column=4)


treeGenerarCurriculo = ttk.Treeview(ventana)
vsb = ttk.Scrollbar(ventana, orient="vertical", command=treeGenerarCurriculo.yview, )
treeGenerarCurriculo.configure(yscrollcommand=vsb.set)
vsb.grid(sticky="W", row=4, column=7, rowspan=2)


treeGenerarCurriculo["columns"] = ("id", "trabajos")
treeGenerarCurriculo.column("#0", width=250)
treeGenerarCurriculo.column("id", width=100)
treeGenerarCurriculo.column("trabajos", width=100)
treeGenerarCurriculo.heading("#0", text="Nombre")
treeGenerarCurriculo.heading("id", text="ID")
treeGenerarCurriculo.heading("trabajos", text="Trabajos")
treeGenerarCurriculo.bind("<Double-1>", on_doubleClick_treeGenerarCurriculo)

treeGenerarCurriculo.grid(row=4, column=4 , columnspan=2)


vacio_label = tk.Label(ventana, text="    ")
vacio_label.grid(row=6, column=2)

proyecto_label = tk.Label(ventana, text="Titulo proyecto:")
proyecto_label.grid(row=7, column=0, sticky="E")
proyecto = tk.Entry(ventana, width=100)
proyecto.grid(row=7, column=1, columnspan=4, sticky="W")

resumen_label = tk.Label(ventana, text="Resumen:")
resumen_label.grid(row=8, column=0, sticky="E")
resumen = tk.Text(ventana, width=100, height=10)
resumen.grid(row=8, column=1, columnspan=4, sticky="W")


botonMontarProyecto = tk.Button(ventana, text="Generar proyecto", command=armarProyecto)
botonMontarProyecto.grid(row=9, column=1)


ventana.mainloop()