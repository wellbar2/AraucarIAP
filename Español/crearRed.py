import os
import networkx as nx
from tqdm import tqdm
import difflib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, PULP_CBC_CMD
import re
import sys


def crearRed(anioInicial, anioFinal, ponderadorAfinidad, ponderadorCoI, evaluadoresXProyecto, proyectoXEvaluadores):
    directorioAtualAbsoluto = os.path.abspath("./")
    # directorioAtualAbsoluto = os.path.abspath("../")

    C = nx.Graph()  ###Grafo con los conflictos
    S = nx.Graph()  ###Grafo con los sin conflictos
    A = nx.Graph()  ###Grafo de asignación

    # modelo = SentenceTransformer('allenai/specter2') #nuevo modelo para la similaridad
    modelo = SentenceTransformer('allenai-specter')  # modelo para la similaridad

    ###### funcción similaridad semantica
    def calcular_similaridade_semantica(texto1, texto2, modelo):
        embeddings1 = modelo.encode(texto1, convert_to_tensor=True)
        embeddings2 = modelo.encode(texto2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity[0][0].item()


    ###### funcción calcula peso bipartido
    def calcularPesoBipartido(E,P,B):
        #E = nx.Graph()
        #P = nx.Graph()
        #B = nx.Graph()


        for proyecto, evaluador in B.edges():
            B.edges[proyecto,evaluador]['pesoCoI'] = 0.0000

            # garantiza la orden de proyectos y evaluadores (a veces el networkx cambia el orden de las aristas)
            if B.nodes[evaluador]['tipo'] == 'Evaluador':
                tempEvaluador = evaluador
            if B.nodes[evaluador]['tipo'] == 'Proyecto':
                tempProyecto = evaluador
            if B.nodes[proyecto]['tipo'] == 'Evaluador':
                tempEvaluador = proyecto
            if B.nodes[proyecto]['tipo'] == 'Proyecto':
                tempProyecto = proyecto

            proyecto = tempProyecto
            evaluador = tempEvaluador

            idsEvaluador = E.nodes[evaluador]['ids']
            idsProyecto = P.nodes[proyecto]['ids']

            idsEvaluador = idsEvaluador.split(',')
            #lista para contar solamente una vez por evaluador con más IDs
            listaConflictos = []
            for idEvaluador in idsEvaluador:
                if idEvaluador in idsProyecto:
                    B.edges[proyecto, evaluador]['pesoCoI'] = B.edges[proyecto,evaluador]['pesoCoI'] + (1 * float(ponderadorAfinidad))
                else:
                    vecinosEvaluador = E.adj[evaluador]
                    for vecinoEvaluador in vecinosEvaluador:
                        idsVecino = E.nodes[vecinoEvaluador]['ids']
                        idsVecino = idsVecino.split(',')
                        for idVecino in idsVecino:
                            if (idVecino in idsProyecto) and (vecinoEvaluador not in listaConflictos):
                                listaConflictos.append(vecinoEvaluador)
                                B.edges[proyecto, evaluador]['pesoCoI'] = B.edges[proyecto, evaluador]['pesoCoI'] + float(E.edges[evaluador,vecinoEvaluador]['weight'])

        return B


    ###### fin calcula peso bipartido






    #############iniciando b-maching

    def solve_wbm(evaluadores_nodes, proyectos_nodes, wt):
        ''' A wrapper function that uses pulp to formulate and solve a WBM'''

        prob = LpProblem("WBM Problem", LpMaximize)

        # Create The Decision variables
        choices = LpVariable.dicts("e", (evaluadores_nodes, proyectos_nodes), 0, 1, LpInteger)

        # Add the objective function
        prob += lpSum([wt[u][v] * choices[u][v]
                       for u in evaluadores_nodes
                       for v in proyectos_nodes]), "Total weights of selected edges"

        # Constraint set ensuring that the total from/to each node
        # is less than its capacity
        for u in evaluadores_nodes:
            for v in proyectos_nodes:
                prob += lpSum([choices[u][v] for v in proyectos_nodes]) <= proyectosCapacidad[v], ""
                prob += lpSum([choices[u][v] for u in evaluadores_nodes]) <= evaluadoresCapacidad[u], ""

        # The problem is solved using PuLP's choice of Solver
        prob.solve(PULP_CBC_CMD(msg=0))

        return (prob)

    def get_selected_edges(prob):

        selected_from = [v.name.split("_")[1:3] for v in prob.variables() if v.value() > 1e-3]
        selected_to = [v.name.split("_")[3:] for v in prob.variables() if v.value() > 1e-3]

        selected_edges = []

        for su, sv in list(zip(selected_from, selected_to)):
            selected_edges.append((su, sv))
        return (selected_edges)

    def create_wt_doubledict(evaluadoresNodes, proyectosNodes):
        wt = {}
        for u in evaluadoresNodes:
            wt[u] = {}
            for v in proyectosNodes:
                wt[u][v] = 0

        for k, val in wts.items():
            u, v = k[0], k[1]
            wt[u][v] = val
        return (wt)

    #############fin b-maching

    def crearDicIdsEvaluadores():
        curriculos = os.listdir(directorioAtualAbsoluto + "//evaluadores//")
        dicIdsEvaluadores = dict()

        # hace el analisis para cada curriculo en el directorio
        for curriculo in curriculos:
            datosCurriculo = open(directorioAtualAbsoluto + "//evaluadores//" + curriculo, "r", encoding="utf8")
            ids = datosCurriculo.readline().replace("\n", "")
            idsLimpios = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            curriculo = curriculo[:-4]
            for registro in idsLimpios.split(","):
                dicIdsEvaluadores.update({registro: curriculo})
            datosCurriculo.close()
        return dicIdsEvaluadores

    def crearRedEvaluadores(anioInicial, anioFinal):
        anioInicial = int(anioInicial)
        anioFinal = int(anioFinal)
        G = nx.Graph()
        curriculos = os.listdir(directorioAtualAbsoluto + "//evaluadores//")
        dicIdsEvaluadores = dict()
        dicIdsEvaluadores = crearDicIdsEvaluadores()

        # hace el analisis para cada curriculo en el directorio
        for curriculo in curriculos:
            datosCurriculo = open(directorioAtualAbsoluto + "//evaluadores//" + curriculo, "r", encoding="utf8")
            ids = datosCurriculo.readline().replace("\n", "")
            idsLimpios = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            curriculo = curriculo[:-4]
            #for registro in idsLimpios.split(","):
                #dicIdsEvaluadores.update({registro: curriculo})

            # crea un node del tipo "Evaluador" para cada curriculo, contiendo todos los IDs
            G.add_node(curriculo, ids=idsLimpios, tipo="Evaluador", trabajos="", anoPrimeroPaper=3000, anoUltimoPaper=0)

            trabajos = ""
            for linea in datosCurriculo:
                coautores = linea.split("||")[3].replace("[", "").replace("]", "").replace("}", "").replace("\n",
                                                                                                            "")  # la lista de coutores de este curriculo sin sucieras
                # cada coautor empieza con "{" y termina con "}" por ser un dict json con más informaciones que necesito
                idCoautores = coautores.split("{")

                ano = int(linea.split("||")[1])
                anioInicialAutor = int(G.nodes[curriculo]["anoPrimeroPaper"])
                anioFinalAutor = int(G.nodes[curriculo]["anoUltimoPaper"])
                if anioInicialAutor > ano:
                    G.nodes[curriculo]["anoPrimeroPaper"] = ano
                if anioFinalAutor < ano:
                    G.nodes[curriculo]["anoUltimoPaper"] = ano

                if (anioInicial <= ano) and (anioFinal >= ano):
                    trabajos = trabajos + "\n" + linea.split("||")[0]

                for id in idCoautores:
                    if (anioInicial > ano) or (anioFinal < ano):  # entre los años elegidos, para obtener periodos
                        continue
                    if len(id) < 6:
                        idCoautores.remove(id)
                    else:
                        id = id.split(",")
                        cardinalidad = len(id)
                        nombreCoautor = id[1][18:-1]  # toma solo el nombre limpio del coautor
                        id = id[0][28:-1]  # toma solo el ID limpio del coautor


                        if id in dicIdsEvaluadores:
                            nombreCoautor = dicIdsEvaluadores[id]


                        if G.has_node(nombreCoautor):
                            anioInicialColaborador = int(G.nodes[nombreCoautor]["anoPrimeroPaper"])
                            anioFinalColaborador = int(G.nodes[nombreCoautor]["anoUltimoPaper"])

                            if anioInicialColaborador > ano:
                                G.nodes[nombreCoautor]["anoPrimeroPaper"] = ano
                            if anioFinalColaborador < ano:
                                G.nodes[nombreCoautor]["anoUltimoPaper"] = ano

                            # si el vertice existe y la aresta tambien suma frecuencia
                            if G.has_edge(curriculo, nombreCoautor):
                                G[curriculo][nombreCoautor]["weight"] = G[curriculo][nombreCoautor]["weight"] + (1 / cardinalidad)
                            else:
                                # si el vertice existe pero no la aresta
                                G.add_edge(curriculo, nombreCoautor, weight=(1 / cardinalidad))

                        else:
                            # si el ID es del Autor no va a crear un nuevo vertice
                            #if id not in (G.nodes[curriculo]["ids"]):
                            # si el vertice es nuevo lo crea
                            G.add_node(nombreCoautor, tipo="Colaborador", ids=id, anoPrimeroPaper=ano, anoUltimoPaper=ano)
                            # además, crea la aresta entre el curriculo actual y el nuevo coautor
                            G.add_edge(curriculo, nombreCoautor, weight=(1 / cardinalidad))



            G.nodes[curriculo]["cifraTrabajos"] = len(trabajos.split("\n"))
            G.nodes[curriculo]["trabajos"] = trabajos

        verticesRemover = []
        for verticeActual in G.nodes():  # remover lo vertices que se repiten con los curriculos
            if G.nodes[verticeActual]["tipo"] != ("Evaluador"):
                idAutor = G.nodes[verticeActual]["ids"]
                if idAutor in dicIdsEvaluadores:
                    VerticeAutorOriginal = dicIdsEvaluadores[idAutor]
                    for vecino in G.neighbors(verticeActual):
                        if G.has_edge(verticeActual, VerticeAutorOriginal):  # si la arista existe
                            if G.has_edge(VerticeAutorOriginal, vecino):
                                # suma el peso de la arista que va a salir al que ya existe
                                G[VerticeAutorOriginal][vecino]['weight'] = G[VerticeAutorOriginal][verticeActual]['weight'] + (1 / cardinalidad)
                            else:
                                #G.add_edge(VerticeAutorOriginal, vecino, weight=0)
                                G.add_edge(VerticeAutorOriginal, vecino, weight=(1 / cardinalidad))

                            G[VerticeAutorOriginal][verticeActual]['weight'] = 0
                        else:
                            # si la no arista existía, la crea con el peso de la arista que va a salir
                            G.add_edge(VerticeAutorOriginal, vecino, weight=G[verticeActual][vecino]['weight'])

                    verticesRemover.append(verticeActual)

        G.remove_nodes_from(verticesRemover)
        G.remove_edges_from(nx.selfloop_edges(G))

        nx.write_gexf(G, directorioAtualAbsoluto + "//redes//evaluadores_" + str(anioInicial) + "_" + str(
            anioFinal) + ".gexf")
        return G  # directorioAtualAbsoluto + "//redes//evaluadores_" + str(anioInicial) + "_" + str(anioFinal) + ".gexf"

    # en verdad crea una base de dados, porque los proyectos no tienen relaciones, pero en formato gefx para los proximos pasos
    def crearRedProyectos(anioInicial, anioFinal):
        G = nx.Graph()
        proyectos = os.listdir(directorioAtualAbsoluto + "//proyectos//")

        for proyecto in proyectos:
            datos = open(directorioAtualAbsoluto + "//proyectos//" + proyecto, "r", encoding="utf8")
            ids = datos.readline().replace("\n", "")
            idsLimpios = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            proyecto = proyecto[:-4]
            resumen = datos.readline()
            # crea un nodo del tipo "proyecto" para cada proyecto, contiendo todos los IDs de autores del proyecto
            G.add_node(proyecto, ids=idsLimpios, tipo="Proyecto", resumen=resumen, idsAutores=idsLimpios)

        nx.write_gexf(G, directorioAtualAbsoluto + "//redes//proyectos_" + str(anioInicial) + "_" + str(
            anioFinal) + ".gexf")
        return G  # directorioAtualAbsoluto + "//redes//proyectos_" + str(anioInicial) + "_" + str(anioFinal) + ".gexf"




    def crearRedBipartida(redEvaluadores, redProyectos):
        E = redEvaluadores
        P = redProyectos
        B = nx.Graph()
        noElegir = []
        removerGradoCero = []

        informe = open(directorioAtualAbsoluto + "//informeConflictos.txt", "w", encoding="UTF8")
        print("Creando el grafo bipartido y lista de ponteciales conflictos")
        totalEvaluadores = [x for x, y in E.nodes(data=True) if y['tipo'] == "Evaluador"]
        print("Total de Evaluadores : " + str(len(totalEvaluadores)))
        print("Total proyectos: " + str(P.number_of_nodes()))
        cont = 0

        for investigador in tqdm(E.nodes()):
            ids = E.nodes[investigador]["ids"]
            ids = ids.split(",")
            for idInvestigador in ids:
                for proyecto in P.nodes():
                    if E.nodes[investigador]["tipo"] == "Evaluador":

                        cifraTrabajos = E.nodes[investigador]["cifraTrabajos"]
                        curriculoInvestigador = E.nodes[investigador]["trabajos"]
                        resumenProyecto = P.nodes[proyecto]["resumen"]


                        B.add_node(investigador, tipo="Evaluador", cifraTrabajos=cifraTrabajos,
                                   trabajos=curriculoInvestigador)
                        B.add_node(proyecto, tipo="Proyecto", resumen=resumenProyecto)


                        if not (B.has_edge(investigador, proyecto)) or B.edges[investigador, proyecto][
                            'PesoSimilaridad'] == 0:
                            cont += 1
                            print(
                                'Calculando ' + str(cont) + ' de ' + str(len(totalEvaluadores * P.number_of_nodes())))

                            ###similaridad con BERT
                            # modelo = SentenceTransformer('allenai-specter')
                            similaridad = calcular_similaridade_semantica(curriculoInvestigador, resumenProyecto,
                                                                          modelo)

                            ###Similaridad con distancia de consenos
                            # similaridad = difflib.SequenceMatcher(None,curriculoInvestigador, resumenProyecto).ratio() #cambiar metodo por mejor

                            similaridad = round(similaridad, 6)
                            pesoCoI = 0.0


                            if B.add_edge(investigador, proyecto):
                                B.edges[investigador, proyecto]['PesoSimilaridad'] = similaridad
                            else:
                                B.add_edge(investigador, proyecto, PesoSimilaridad=similaridad, pesoCoI=float(pesoCoI))


                    autoresProyecto = P.nodes[proyecto]["ids"]
                    dobleRol = False


                    if idInvestigador in autoresProyecto:  # pq si ha entrado en la red es coautor de alguno ego de la red egocentrica
                        if E.nodes[investigador]["tipo"] == "Evaluador":
                            informe.writelines("Actor con doble rol: " + investigador + "\n")
                            C.add_node(investigador, tipo="DobleRoll")
                            C.add_node(proyecto, tipo="Proyecto")
                            C.add_edge(investigador, proyecto, pesoCoI=float((1 * float(ponderadorAfinidad))))
                            B.edges[investigador, proyecto]["pesoCoI"] = B.edges[investigador, proyecto]["pesoCoI"] + (
                                        1 * float(ponderadorAfinidad))  # garantiza borrar la arista doble roll
                            noElegir.append(investigador)
                            dobleRol = True


                            ####doble rol
                            vecinos = E.adj[investigador]
                            for vecino in vecinos:
                                if E.nodes[vecino]["tipo"] == "Evaluador":
                                    # peso fraccionado  /\ E.nodes[vecino]["cifraTrabajos"] #peso Normalizado
                                    pesoCoI = E.edges[investigador, vecino]["weight"]
                                    pesoCoI = round(pesoCoI, 6)
                                    informe.writelines(
                                        vecino + " - Evaluador con relación directa! Relación con: " + investigador + " >>>>>> " + proyecto + "\n")


                                    if B.has_edge(vecino, proyecto):
                                        B.edges[vecino, proyecto]["pesoCoI"] = B.edges[vecino, proyecto][
                                                                                   "pesoCoI"] + float(pesoCoI)
                                    else:
                                        B.add_edge(vecino, proyecto, pesoCoI=float(pesoCoI), PesoSimilaridad=float(0))


                                    if C.has_node(investigador):
                                        categoria = C.nodes[investigador]["tipo"]
                                        C.nodes[investigador]["tipo"] = categoria + "/" + "Autor"
                                    else:
                                        C.add_node(investigador, tipo="Autor")

                                    if C.has_node(vecino):
                                        categoria = C.nodes[vecino]["tipo"]
                                        C.nodes[vecino]["tipo"] = categoria + "/" + "RelacionDirecta"
                                    else:
                                        C.add_node(vecino, tipo="RelacionDirecta")

                                    C.add_node(proyecto, tipo="Proyecto")

                                    pesoCoI = pesoCoI + (1 * float(
                                        ponderadorAfinidad))  # si es doble rol suma 1 para garantizar que la arista se va a borrar

                                    if C.has_edge(investigador, proyecto):
                                        C.edges[investigador, proyecto]["pesoCoI"] = C.edges[investigador, proyecto][
                                                                                         "pesoCoI"] + pesoCoI
                                    else:
                                        C.add_edge(investigador, proyecto, pesoCoI=float(pesoCoI))

                                    if C.has_edge(investigador, vecino):
                                        C.edges[investigador, vecino]["pesoCoI"] = C.edges[investigador, vecino][
                                                                                       "pesoCoI"] + pesoCoI
                                    else:
                                        C.add_edge(investigador, vecino, pesoCoI=float(pesoCoI))


                                    if B.has_edge(vecino, proyecto):
                                        B.edges[vecino, proyecto]["pesoCoI"] = B.edges[vecino, proyecto][
                                                                                   "pesoCoI"] + float(pesoCoI)
                                    else:
                                        B.add_edge(vecino, proyecto, pesoCoI=float(pesoCoI),
                                                   PesoSimilaridad=float(0))


                                    noElegir.append(vecino)
                        ####fin doble rol

                        else:
                            vecinos = E.adj[investigador]
                            for vecino in vecinos:
                                if E.nodes[vecino]["tipo"] == "Evaluador":
                                    pesoCoI = E.edges[investigador, vecino][
                                        "weight"]  # peso fraccionado  / E.nodes[vecino]["cifraTrabajos"] #peso Normalizado
                                    pesoCoI = round(pesoCoI, 6)
                                    informe.writelines(
                                        vecino + " - Evaluador con relación directa! Relación con: " + investigador + " >>>>>> " + proyecto + "\n")


                                    if C.has_node(investigador):
                                        categoria = C.nodes[investigador]["tipo"]
                                        C.nodes[investigador]["tipo"] = categoria + "/" + "Autor"
                                    else:
                                        C.add_node(investigador, tipo="Autor")

                                    if C.has_node(vecino):
                                        categoria = C.nodes[vecino]["tipo"]
                                        C.nodes[vecino]["tipo"] = categoria + "/" + "RelacionDirecta"
                                    else:
                                        C.add_node(vecino, tipo="RelacionDirecta")

                                    C.add_node(proyecto, tipo="Proyecto")

                                    if C.has_edge(investigador, proyecto):
                                        C.edges[investigador, proyecto]["pesoCoI"] = C.edges[investigador, proyecto][
                                                                                         "pesoCoI"] + pesoCoI
                                    else:
                                        C.add_edge(investigador, proyecto, pesoCoI=float(pesoCoI))

                                    if C.has_edge(investigador, vecino):
                                        C.edges[investigador, vecino]["pesoCoI"] = C.edges[investigador, vecino][
                                                                                       "pesoCoI"] + pesoCoI
                                    else:
                                        C.add_edge(investigador, vecino, pesoCoI=float(pesoCoI))

                                    if B.has_edge(vecino, proyecto):
                                        B.edges[vecino, proyecto]["pesoCoI"] = B.edges[vecino, proyecto][
                                                                                   "pesoCoI"] + float(pesoCoI)
                                    else:
                                        B.add_edge(vecino, proyecto, pesoCoI=float(pesoCoI), PesoSimilaridad=float(0))

                                    noElegir.append(vecino)


        B = calcularPesoBipartido(E, P, B)


        informe.writelines("\n\n####Evaluadores con CoI####\n")
        sinConflicto = [x for x, y in E.nodes(data=True) if y["tipo"] == "Evaluador"]
        noElegir = list(dict.fromkeys(noElegir))
        for CoI in noElegir:
            informe.writelines(CoI + "\n")
            sinConflicto.remove(CoI)

        informe.writelines("\n\n####Evaluadores sin CoI####\n")
        S.add_nodes_from(sinConflicto)
        for SCoI in sinConflicto:
            informe.writelines(SCoI + "\n")

        for origen, destino in B.edges:
            similaridadPonderada = float(B.edges[origen, destino]["PesoSimilaridad"]) * float(ponderadorAfinidad)
            CoIPonderado = float(B.edges[origen, destino]["pesoCoI"]) * float(ponderadorCoI)
            indAtrib = (similaridadPonderada - CoIPonderado) / (float(ponderadorAfinidad))

            if indAtrib >= 0:
                B.edges[origen, destino]["indAtrib"] = indAtrib
            else:
                #B.edges[origen, destino]["indAtrib"] = 0 #mantiene los IAP < 0
                B.remove_edge(origen, destino)


                if B.degree[origen] == 0:
                    removerGradoCero.append(origen)
                    print(origen + " removido del grafo")
                if B.degree[destino] == 0:
                    removerGradoCero.append(destino)
                    print(destino + " removido del grafo")

        if len(removerGradoCero) >= 1:
            B.remove_nodes_from(removerGradoCero)
            gradoCero = open("gradoCero.txt", 'w', encoding='utf8')
            for nodo in removerGradoCero:
                gradoCero.writelines(nodo + '\n')
            gradoCero.close()

        return B


    grafoEvaluadores = crearRedEvaluadores(anioInicial, anioFinal)
    grafoProyectos = crearRedProyectos(anioInicial, anioFinal)

    G = crearRedBipartida(grafoEvaluadores, grafoProyectos)


    nx.write_gexf(C, directorioAtualAbsoluto + "//redes//conflictos.gexf")
    nx.write_gexf(G, directorioAtualAbsoluto + "//redes//bipartido.gexf")

    evaluadoresNodes = []
    for node in G.nodes:
        if G.nodes[node]["tipo"] == "Evaluador":
            evaluadoresNodes.append(node)

    proyectosNodes = []
    for node in G.nodes:
        if G.nodes[node]["tipo"] == "Proyecto":
            proyectosNodes.append(node)

    evaluadoresCapacidad = {}  # v node capacities
    for evaluador in evaluadoresNodes:
        evaluadoresCapacidad.update({evaluador: int(proyectoXEvaluadores)})

    proyectosCapacidad = {}  # u node capacities
    for proyecto in proyectosNodes:
        proyectosCapacidad.update({proyecto: int(evaluadoresXProyecto)})

    wts = {}
    for edge in G.edges:
        # evaluador siempre debe estar a la izquierda
        if G.nodes[edge[0]]["tipo"] == "Evaluador":
            izquierda = edge[0]
            derecha = edge[1]
        else:
            izquierda = edge[1]
            derecha = edge[0]
        edgeOrdenado = (izquierda, derecha)
        wts.update({edgeOrdenado: G.edges[edge]["indAtrib"]})

    wt = create_wt_doubledict(evaluadoresNodes, proyectosNodes)
    p = solve_wbm(evaluadoresNodes, proyectosNodes, wt)

    selected_edges = get_selected_edges(p)
    listaOrdenada = open("listaOrdenada.txt", "w")

    for sugerencia in selected_edges:
        for nodo in G.nodes:
            primero = re.sub(u'[^a-zA-Z0-9]', '', str(sugerencia[0]))
            segundo = re.sub(u'[^a-zA-Z0-9]', '', nodo)
            if G.nodes[nodo]["tipo"] == "Evaluador":
                if primero == segundo:
                    flagEvaluador = True
                else:
                    flagEvaluador = False

                if flagEvaluador:
                    nombreSugerencia = nodo

            primero = re.sub(u'[^a-zA-Z0-9]', '', str(sugerencia[1]))
            segundo = re.sub(u'[^a-zA-Z0-9]', '', nodo)
            if G.nodes[nodo]["tipo"] == "Proyecto":
                if primero == segundo:
                    flagProyecto = True
                else:
                    flagProyecto = False

                if flagProyecto:
                    nombreProyecto = nodo

        listaOrdenada.writelines("Evaluador: " + str(nombreSugerencia) + " >>>>>>> trabajo: " + str(nombreProyecto) + " >>>>>>> peso indicador de atribuição: " + str(G.edges[nombreSugerencia, nombreProyecto]["indAtrib"]) + "\n")

        print("Evaluador: " + str(nombreSugerencia) + " >>>>>>> trabajo: " + str(nombreProyecto) + " >>>>>>> peso indicador de atribuição: " + str(G.edges[nombreSugerencia, nombreProyecto]["indAtrib"]))

        A.add_node(nombreSugerencia, trabajos=G.nodes[nombreSugerencia]["trabajos"])
        A.add_node(nombreProyecto, resumen=G.nodes[nombreProyecto]["resumen"])
        A.add_edge(nombreSugerencia, nombreProyecto)
        A.edges[nombreSugerencia, nombreProyecto]['IAP'] = G.edges[nombreSugerencia, nombreProyecto]["indAtrib"]

    nx.write_gexf(A, directorioAtualAbsoluto + "//redes//A.gexf")

    Z = nx.Graph()

    Z = nx.compose(S, C)

    nx.write_gexf(Z, directorioAtualAbsoluto + "//redes//ConflictosXSinConflictos.gexf")

    print('Gracias por usar nuestra aplicación')


def crear():
    crearRed(anioInicial_param, anioFinal_param, ponderadorAfinidad_param, ponderadorCoI_param,
             evaluadoresXProyecto_param, proyectoXEvaluadores_param)


if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Número incorreto de parâmetros.")
        sys.exit(1)

    anioInicial_param = sys.argv[1]
    anioFinal_param = sys.argv[2]
    ponderadorAfinidad_param = sys.argv[3]
    ponderadorCoI_param = sys.argv[4]
    evaluadoresXProyecto_param = sys.argv[5]
    proyectoXEvaluadores_param = sys.argv[6]

    crearRed(
        anioInicial_param,
        anioFinal_param,
        ponderadorAfinidad_param,
        ponderadorCoI_param,
        evaluadoresXProyecto_param,
        proyectoXEvaluadores_param
    )

