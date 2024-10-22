import os
import networkx as nx
from tqdm import tqdm
import difflib
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer, util
from pulp import LpProblem, LpMaximize, LpVariable, lpSum, LpInteger, PULP_CBC_CMD
import re
import sys

def createNetwork(startYear, endYear, affinityWeight, coiWeight, evaluatorsPerProject, projectsPerEvaluator):
    absoluteCurrentDirectory = os.path.abspath("./")
    # absoluteCurrentDirectory = os.path.abspath("../")

    C = nx.Graph()  # Graph with conflicts
    S = nx.Graph()  # Graph without conflicts
    A = nx.Graph()  # Assignment graph

    # model = SentenceTransformer('allenai/specter2')  # New model for similarity
    model = SentenceTransformer('allenai-specter')  # Model for similarity

    ###### Semantic similarity function
    def calculate_semantic_similarity(text1, text2, model):
        embeddings1 = model.encode(text1, convert_to_tensor=True)
        embeddings2 = model.encode(text2, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings1, embeddings2)
        return similarity[0][0].item()

    ###### Function to calculate bipartite weight
    def calculateBipartiteWeight(E, P, B):
        # E = nx.Graph()
        # P = nx.Graph()
        # B = nx.Graph()

        for project, evaluator in B.edges():
            B.edges[project, evaluator]['coiWeight'] = 0.0000

            # Ensures the order of projects and evaluators (sometimes networkx changes the order of edges)
            if B.nodes[evaluator]['type'] == 'Evaluator':
                tempEvaluator = evaluator
            if B.nodes[evaluator]['type'] == 'Project':
                tempProject = evaluator
            if B.nodes[project]['type'] == 'Evaluator':
                tempEvaluator = project
            if B.nodes[project]['type'] == 'Project':
                tempProject = project

            project = tempProject
            evaluator = tempEvaluator

            evaluatorIds = E.nodes[evaluator]['ids']
            projectIds = P.nodes[project]['ids']

            evaluatorIds = evaluatorIds.split(',')
            # List to count only once per evaluator with more IDs
            conflictList = []
            for evaluatorId in evaluatorIds:
                if evaluatorId in projectIds:
                    B.edges[project, evaluator]['coiWeight'] += (1 * float(affinityWeight))
                else:
                    evaluatorNeighbors = E.adj[evaluator]
                    for evaluatorNeighbor in evaluatorNeighbors:
                        neighborIds = E.nodes[evaluatorNeighbor]['ids']
                        neighborIds = neighborIds.split(',')
                        for neighborId in neighborIds:
                            if (neighborId in projectIds) and (evaluatorNeighbor not in conflictList):
                                conflictList.append(evaluatorNeighbor)
                                B.edges[project, evaluator]['coiWeight'] += float(E.edges[evaluator, evaluatorNeighbor]['weight'])

        return B

    ###### End of bipartite weight calculation

    ############# Starting b-matching

    def solve_wbm(evaluator_nodes, project_nodes, wt):
        '''A wrapper function that uses PuLP to formulate and solve a WBM'''

        prob = LpProblem("WBM Problem", LpMaximize)

        # Create the decision variables
        choices = LpVariable.dicts("e", (evaluator_nodes, project_nodes), 0, 1, LpInteger)

        # Add the objective function
        prob += lpSum([wt[u][v] * choices[u][v]
                       for u in evaluator_nodes
                       for v in project_nodes]), "Total weights of selected edges"

        # Constraint set ensuring that the total from/to each node
        # is less than its capacity
        for u in evaluator_nodes:
            for v in project_nodes:
                prob += lpSum([choices[u][v] for v in project_nodes]) <= projectCapacities[v], ""
                prob += lpSum([choices[u][v] for u in evaluator_nodes]) <= evaluatorCapacities[u], ""

        # The problem is solved using PuLP's choice of Solver
        prob.solve(PULP_CBC_CMD(msg=0))

        return prob

    def get_selected_edges(prob):
        selected_from = [v.name.split("_")[1:3] for v in prob.variables() if v.value() > 1e-3]
        selected_to = [v.name.split("_")[3:] for v in prob.variables() if v.value() > 1e-3]

        selected_edges = []

        for su, sv in list(zip(selected_from, selected_to)):
            selected_edges.append((su, sv))
        return selected_edges

    def create_wt_doubledict(evaluatorNodes, projectNodes):
        wt = {}
        for u in evaluatorNodes:
            wt[u] = {}
            for v in projectNodes:
                wt[u][v] = 0

        for k, val in wts.items():
            u, v = k[0], k[1]
            wt[u][v] = val
        return wt

    ############# End of b-matching

    def createEvaluatorIdsDict():
        resumes = os.listdir(absoluteCurrentDirectory + "//evaluators//")
        evaluatorIdsDict = dict()

        # Analyzes each resume in the directory
        for resume in resumes:
            resumeData = open(absoluteCurrentDirectory + "//evaluators//" + resume, "r", encoding="utf8")
            ids = resumeData.readline().replace("\n", "")
            cleanIds = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            resume = resume[:-4]
            for record in cleanIds.split(","):
                evaluatorIdsDict.update({record: resume})
            resumeData.close()
        return evaluatorIdsDict

    def createEvaluatorNetwork(startYear, endYear):
        startYear = int(startYear)
        endYear = int(endYear)
        G = nx.Graph()
        resumes = os.listdir(absoluteCurrentDirectory + "//evaluators//")
        evaluatorIdsDict = dict()
        evaluatorIdsDict = createEvaluatorIdsDict()

        # Analyzes each resume in the directory
        for resume in resumes:
            resumeData = open(absoluteCurrentDirectory + "//evaluators//" + resume, "r", encoding="utf8")
            ids = resumeData.readline().replace("\n", "")
            cleanIds = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            resume = resume[:-4]
            # Creates a node of type "Evaluator" for each resume, containing all IDs
            G.add_node(resume, ids=cleanIds, type="Evaluator", works="", firstPaperYear=3000, lastPaperYear=0)

            works = ""
            for line in resumeData:
                coauthors = line.split("||")[3].replace("[", "").replace("]", "").replace("}", "").replace("\n", "")
                # Each coauthor starts with "{" and ends with "}" as it's a JSON dict with more info than needed
                coauthorIds = coauthors.split("{")

                year = int(line.split("||")[1])
                authorStartYear = int(G.nodes[resume]["firstPaperYear"])
                authorEndYear = int(G.nodes[resume]["lastPaperYear"])
                if authorStartYear > year:
                    G.nodes[resume]["firstPaperYear"] = year
                if authorEndYear < year:
                    G.nodes[resume]["lastPaperYear"] = year

                if (startYear <= year) and (endYear >= year):
                    works = works + "\n" + line.split("||")[0]

                for id in coauthorIds:
                    if (startYear > year) or (endYear < year):  # Between the chosen years to get periods
                        continue
                    if len(id) < 6:
                        coauthorIds.remove(id)
                    else:
                        id = id.split(",")
                        cardinality = len(id)
                        coauthorName = id[1][18:-1]  # Extracts clean coauthor name
                        id = id[0][28:-1]  # Extracts clean coauthor ID

                        if id in evaluatorIdsDict:
                            coauthorName = evaluatorIdsDict[id]

                        if G.has_node(coauthorName):
                            collaboratorStartYear = int(G.nodes[coauthorName]["firstPaperYear"])
                            collaboratorEndYear = int(G.nodes[coauthorName]["lastPaperYear"])

                            if collaboratorStartYear > year:
                                G.nodes[coauthorName]["firstPaperYear"] = year
                            if collaboratorEndYear < year:
                                G.nodes[coauthorName]["lastPaperYear"] = year

                            # If the node and edge exist, increase frequency
                            if G.has_edge(resume, coauthorName):
                                G[resume][coauthorName]["weight"] += (1 / cardinality)
                            else:
                                # If the node exists but not the edge
                                G.add_edge(resume, coauthorName, weight=(1 / cardinality))
                        else:
                            # If the ID is of the author, don't create a new node
                            # if id not in (G.nodes[resume]["ids"]):
                            # If the node is new, create it
                            G.add_node(coauthorName, type="Collaborator", ids=id, firstPaperYear=year, lastPaperYear=year)
                            # Also, create the edge between the current resume and the new coauthor
                            G.add_edge(resume, coauthorName, weight=(1 / cardinality))

            G.nodes[resume]["numWorks"] = len(works.split("\n"))
            G.nodes[resume]["works"] = works

        nodesToRemove = []
        for currentNode in G.nodes():  # Remove nodes that repeat with the resumes
            if G.nodes[currentNode]["type"] != "Evaluator":
                authorId = G.nodes[currentNode]["ids"]
                if authorId in evaluatorIdsDict:
                    originalAuthorNode = evaluatorIdsDict[authorId]
                    for neighbor in G.neighbors(currentNode):
                        if G.has_edge(currentNode, originalAuthorNode):  # If the edge exists
                            if G.has_edge(originalAuthorNode, neighbor):
                                # Sum the weight of the edge to be removed to the existing one
                                G[originalAuthorNode][neighbor]['weight'] += G[originalAuthorNode][currentNode]['weight'] + (1 / cardinality)
                            else:
                                # G.add_edge(originalAuthorNode, neighbor, weight=0)
                                G.add_edge(originalAuthorNode, neighbor, weight=(1 / cardinality))

                            G[originalAuthorNode][currentNode]['weight'] = 0
                        else:
                            # If the edge didn't exist, create it with the weight of the edge to be removed
                            G.add_edge(originalAuthorNode, neighbor, weight=G[currentNode][neighbor]['weight'])

                    nodesToRemove.append(currentNode)

        G.remove_nodes_from(nodesToRemove)
        G.remove_edges_from(nx.selfloop_edges(G))

        nx.write_gexf(G, absoluteCurrentDirectory + "//networks//evaluators_" + str(startYear) + "_" + str(endYear) + ".gexf")
        return G  # absoluteCurrentDirectory + "//networks//evaluators_" + str(startYear) + "_" + str(endYear) + ".gexf"

    # Actually creates a database, since projects don't have relationships, but in gefx format for the next steps
    def createProjectNetwork(startYear, endYear):
        G = nx.Graph()
        projects = os.listdir(absoluteCurrentDirectory + "//projects//")

        for project in projects:
            data = open(absoluteCurrentDirectory + "//projects//" + project, "r", encoding="utf8")
            ids = data.readline().replace("\n", "")
            cleanIds = ids.replace("[", "").replace("'", "").replace("]", "").replace(" ", "")
            project = project[:-4]
            summary = data.readline()
            # Creates a node of type "Project" for each project, containing all author IDs of the project
            G.add_node(project, ids=cleanIds, type="Project", summary=summary, authorIds=cleanIds)

        nx.write_gexf(G, absoluteCurrentDirectory + "//networks//projects_" + str(startYear) + "_" + str(endYear) + ".gexf")
        return G  # absoluteCurrentDirectory + "//networks//projects_" + str(startYear) + "_" + str(endYear) + ".gexf"

    def createBipartiteNetwork(evaluatorNetwork, projectNetwork):
        E = evaluatorNetwork
        P = projectNetwork
        B = nx.Graph()
        doNotSelect = []
        removeZeroDegree = []

        report = open(absoluteCurrentDirectory + "//conflictReport.txt", "w", encoding="UTF8")
        print("Creating the bipartite graph and potential conflict list")
        totalEvaluators = [x for x, y in E.nodes(data=True) if y['type'] == "Evaluator"]
        print("Total Evaluators: " + str(len(totalEvaluators)))
        print("Total Projects: " + str(P.number_of_nodes()))
        count = 0

        for researcher in tqdm(E.nodes()):
            ids = E.nodes[researcher]["ids"]
            ids = ids.split(",")
            for researcherId in ids:
                for project in P.nodes():
                    if E.nodes[researcher]["type"] == "Evaluator":

                        numWorks = E.nodes[researcher]["numWorks"]
                        researcherResume = E.nodes[researcher]["works"]
                        projectSummary = P.nodes[project]["summary"]

                        B.add_node(researcher, type="Evaluator", numWorks=numWorks, works=researcherResume)
                        B.add_node(project, type="Project", summary=projectSummary)

                        if not (B.has_edge(researcher, project)) or B.edges[researcher, project]['SimilarityWeight'] == 0:
                            count += 1
                            print('Calculating ' + str(count) + ' of ' + str(len(totalEvaluators * P.number_of_nodes())))

                            ### Similarity with BERT
                            # model = SentenceTransformer('allenai-specter')
                            similarity = calculate_semantic_similarity(researcherResume, projectSummary, model)

                            ### Similarity with cosine distance
                            # similarity = difflib.SequenceMatcher(None, researcherResume, projectSummary).ratio()  # Change method for better one

                            similarity = round(similarity, 6)
                            coiWeight = 0.0

                            if B.add_edge(researcher, project):
                                B.edges[researcher, project]['SimilarityWeight'] = similarity
                            else:
                                B.add_edge(researcher, project, SimilarityWeight=similarity, coiWeight=float(coiWeight))

                    projectAuthors = P.nodes[project]["ids"]
                    doubleRole = False

                    if researcherId in projectAuthors:  # Because if entered the network, they are a coauthor in the ego network
                        if E.nodes[researcher]["type"] == "Evaluator":
                            report.writelines("Actor with double role: " + researcher + "\n")
                            C.add_node(researcher, type="DoubleRole")
                            C.add_node(project, type="Project")
                            C.add_edge(researcher, project, coiWeight=float((1 * float(affinityWeight))))
                            B.edges[researcher, project]["coiWeight"] += (1 * float(affinityWeight))  # Ensures the double role edge is removed
                            doNotSelect.append(researcher)
                            doubleRole = True

                            #### Double role
                            neighbors = E.adj[researcher]
                            for neighbor in neighbors:
                                if E.nodes[neighbor]["type"] == "Evaluator":
                                    # Fractionated weight  # / E.nodes[neighbor]["numWorks"]  # Normalized weight
                                    coiWeight = E.edges[researcher, neighbor]["weight"]
                                    coiWeight = round(coiWeight, 6)
                                    report.writelines(
                                        neighbor + " - Evaluator with direct relationship! Relationship with: " + researcher + " >>>>>> " + project + "\n")

                                    if B.has_edge(neighbor, project):
                                        B.edges[neighbor, project]["coiWeight"] += float(coiWeight)
                                    else:
                                        B.add_edge(neighbor, project, coiWeight=float(coiWeight), SimilarityWeight=float(0))

                                    if C.has_node(researcher):
                                        category = C.nodes[researcher]["type"]
                                        C.nodes[researcher]["type"] = category + "/" + "Author"
                                    else:
                                        C.add_node(researcher, type="Author")

                                    if C.has_node(neighbor):
                                        category = C.nodes[neighbor]["type"]
                                        C.nodes[neighbor]["type"] = category + "/" + "DirectRelationship"
                                    else:
                                        C.add_node(neighbor, type="DirectRelationship")

                                    C.add_node(project, type="Project")

                                    coiWeight += (1 * float(
                                        affinityWeight))  # If double role, adds 1 to ensure the edge is removed

                                    if C.has_edge(researcher, project):
                                        C.edges[researcher, project]["coiWeight"] += coiWeight
                                    else:
                                        C.add_edge(researcher, project, coiWeight=float(coiWeight))

                                    if C.has_edge(researcher, neighbor):
                                        C.edges[researcher, neighbor]["coiWeight"] += coiWeight
                                    else:
                                        C.add_edge(researcher, neighbor, coiWeight=float(coiWeight))

                                    if B.has_edge(neighbor, project):
                                        B.edges[neighbor, project]["coiWeight"] += float(coiWeight)
                                    else:
                                        B.add_edge(neighbor, project, coiWeight=float(coiWeight), SimilarityWeight=float(0))

                                    doNotSelect.append(neighbor)
                            #### End of double role

                        else:
                            neighbors = E.adj[researcher]
                            for neighbor in neighbors:
                                if E.nodes[neighbor]["type"] == "Evaluator":
                                    coiWeight = E.edges[researcher, neighbor]["weight"]  # Fractionated weight  # / E.nodes[neighbor]["numWorks"]  # Normalized weight
                                    coiWeight = round(coiWeight, 6)
                                    report.writelines(
                                        neighbor + " - Evaluator with direct relationship! Relationship with: " + researcher + " >>>>>> " + project + "\n")

                                    if C.has_node(researcher):
                                        category = C.nodes[researcher]["type"]
                                        C.nodes[researcher]["type"] = category + "/" + "Author"
                                    else:
                                        C.add_node(researcher, type="Author")

                                    if C.has_node(neighbor):
                                        category = C.nodes[neighbor]["type"]
                                        C.nodes[neighbor]["type"] = category + "/" + "DirectRelationship"
                                    else:
                                        C.add_node(neighbor, type="DirectRelationship")

                                    C.add_node(project, type="Project")

                                    if C.has_edge(researcher, project):
                                        C.edges[researcher, project]["coiWeight"] += coiWeight
                                    else:
                                        C.add_edge(researcher, project, coiWeight=float(coiWeight))

                                    if C.has_edge(researcher, neighbor):
                                        C.edges[researcher, neighbor]["coiWeight"] += coiWeight
                                    else:
                                        C.add_edge(researcher, neighbor, coiWeight=float(coiWeight))

                                    if B.has_edge(neighbor, project):
                                        B.edges[neighbor, project]["coiWeight"] += float(coiWeight)
                                    else:
                                        B.add_edge(neighbor, project, coiWeight=float(coiWeight), SimilarityWeight=float(0))

                                    doNotSelect.append(neighbor)

        B = calculateBipartiteWeight(E, P, B)

        report.writelines("\n\n#### Evaluators with CoI ####\n")
        withoutConflict = [x for x, y in E.nodes(data=True) if y["type"] == "Evaluator"]
        doNotSelect = list(dict.fromkeys(doNotSelect))
        for CoI in doNotSelect:
            report.writelines(CoI + "\n")
            withoutConflict.remove(CoI)

        report.writelines("\n\n#### Evaluators without CoI ####\n")
        S.add_nodes_from(withoutConflict)
        for SCoI in withoutConflict:
            report.writelines(SCoI + "\n")

        for origin, destination in B.edges:
            weightedSimilarity = float(B.edges[origin, destination]["SimilarityWeight"]) * float(affinityWeight)
            weightedCoI = float(B.edges[origin, destination]["coiWeight"]) * float(coiWeight)
            attributionIndex = (weightedSimilarity - weightedCoI) / (float(affinityWeight))

            if attributionIndex >= 0:
                B.edges[origin, destination]["attributionIndex"] = attributionIndex
            else:
                # B.edges[origin, destination]["attributionIndex"] = 0  # Keeps IAP < 0
                B.remove_edge(origin, destination)

                if B.degree[origin] == 0:
                    removeZeroDegree.append(origin)
                    print(origin + " removed from the graph")
                if B.degree[destination] == 0:
                    removeZeroDegree.append(destination)
                    print(destination + " removed from the graph")

        if len(removeZeroDegree) >= 1:
            B.remove_nodes_from(removeZeroDegree)
            zeroDegree = open("zeroDegree.txt", 'w', encoding='utf8')
            for node in removeZeroDegree:
                zeroDegree.writelines(node + '\n')
            zeroDegree.close()

        return B

    evaluatorGraph = createEvaluatorNetwork(startYear, endYear)
    projectGraph = createProjectNetwork(startYear, endYear)

    G = createBipartiteNetwork(evaluatorGraph, projectGraph)

    nx.write_gexf(C, absoluteCurrentDirectory + "//networks//conflicts.gexf")
    nx.write_gexf(G, absoluteCurrentDirectory + "//networks//bipartite.gexf")

    evaluatorNodes = []
    for node in G.nodes:
        if G.nodes[node]["type"] == "Evaluator":
            evaluatorNodes.append(node)

    projectNodes = []
    for node in G.nodes:
        if G.nodes[node]["type"] == "Project":
            projectNodes.append(node)

    evaluatorCapacities = {}  # v node capacities
    for evaluator in evaluatorNodes:
        evaluatorCapacities.update({evaluator: int(projectsPerEvaluator)})

    projectCapacities = {}  # u node capacities
    for project in projectNodes:
        projectCapacities.update({project: int(evaluatorsPerProject)})

    wts = {}
    for edge in G.edges:
        # Evaluator must always be on the left
        if G.nodes[edge[0]]["type"] == "Evaluator":
            left = edge[0]
            right = edge[1]
        else:
            left = edge[1]
            right = edge[0]
        orderedEdge = (left, right)
        wts.update({orderedEdge: G.edges[edge]["attributionIndex"]})

    wt = create_wt_doubledict(evaluatorNodes, projectNodes)
    p = solve_wbm(evaluatorNodes, projectNodes, wt)

    selected_edges = get_selected_edges(p)
    orderedList = open("orderedList.txt", "w")

    for suggestion in selected_edges:
        for node in G.nodes:
            first = re.sub(u'[^a-zA-Z0-9]', '', str(suggestion[0]))
            second = re.sub(u'[^a-zA-Z0-9]', '', node)
            if G.nodes[node]["type"] == "Evaluator":
                flagEvaluator = first == second
                if flagEvaluator:
                    suggestionName = node

            first = re.sub(u'[^a-zA-Z0-9]', '', str(suggestion[1]))
            second = re.sub(u'[^a-zA-Z0-9]', '', node)
            if G.nodes[node]["type"] == "Project":
                flagProject = first == second
                if flagProject:
                    projectName = node

        orderedList.writelines("Evaluator: " + str(suggestionName) + " >>>>>>> Project: " + str(projectName) + " >>>>>>> Attribution Index Weight: " + str(G.edges[suggestionName, projectName]["attributionIndex"]) + "\n")

        print("Evaluator: " + str(suggestionName) + " >>>>>>> Project: " + str(projectName) + " >>>>>>> Attribution Index Weight: " + str(G.edges[suggestionName, projectName]["attributionIndex"]))

        A.add_node(suggestionName, works=G.nodes[suggestionName]["works"])
        A.add_node(projectName, summary=G.nodes[projectName]["summary"])
        A.add_edge(suggestionName, projectName)
        A.edges[suggestionName, projectName]['IAP'] = G.edges[suggestionName, projectName]["attributionIndex"]

    nx.write_gexf(A, absoluteCurrentDirectory + "//networks//A.gexf")

    Z = nx.Graph()

    Z = nx.compose(S, C)

    nx.write_gexf(Z, absoluteCurrentDirectory + "//networks//ConflictsXWithoutConflicts.gexf")

    print('Thank you for using our application')

def create():
    createNetwork(startYear_param, endYear_param, affinityWeight_param, coiWeight_param,
                  evaluatorsPerProject_param, projectsPerEvaluator_param)

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Incorrect number of parameters.")
        sys.exit(1)

    startYear_param = sys.argv[1]
    endYear_param = sys.argv[2]
    affinityWeight_param = sys.argv[3]
    coiWeight_param = sys.argv[4]
    evaluatorsPerProject_param = sys.argv[5]
    projectsPerEvaluator_param = sys.argv[6]

    createNetwork(
        startYear_param,
        endYear_param,
        affinityWeight_param,
        coiWeight_param,
        evaluatorsPerProject_param,
        projectsPerEvaluator_param
    )