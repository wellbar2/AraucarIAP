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
# from bertopic import BERTopic
# from sklearn.feature_extraction.text import CountVectorizer
# import nltk

def searchAuthorIds(firstName, lastName):
    authorSearch = requests.get(
        "https://api.openalex.org/authors?search=" + firstName + "%20" + lastName + "&per-page=200"
    )
    results = json.loads(authorSearch.text)
    authors = results["results"]

    openAlexIds = []
    for author in authors:
        if re.match(
            unidecode.unidecode("^" + firstName + ".*" + lastName),
            unidecode.unidecode(author["display_name"]),
            re.IGNORECASE,
        ):
            openAlexIds.append(author)

        for item in treeSelectName.get_children():
            treeSelectName.delete(item)

    return openAlexIds

def buildResume():
    absoluteCurrentDirectory = os.path.abspath("./")
    #absoluteCurrentDirectory = os.path.abspath("../")
    directory = absoluteCurrentDirectory + "//evaluators//"
    fileFirstName = firstNameEntry.get()
    fileLastName = lastNameEntry.get()

    if not fileFirstName.strip():
        fileFirstName = '1'
    if not fileLastName.strip():
        fileLastName = '1'

    fileFirstName = re.sub(r'[^a-zA-Z0-9 ]+', '', fileFirstName)
    fileLastName = re.sub(r'[^a-zA-Z0-9 ]+', '', fileLastName)

    file = open(directory + fileFirstName + "-" + fileLastName + ".txt", "w", encoding="utf8")

    authorIds = []
    workCounts = []

    firstNameEntry.delete(0, tkinter.END)
    lastNameEntry.delete(0, tkinter.END)

    for item in treeSelectName.get_children():
        treeSelectName.delete(item)

    for item in treeGenerateResume.get_children():
        authorIds.append(treeGenerateResume.item(item)["values"][0])
        workCounts.append(treeGenerateResume.item(item)["values"][1])
        treeGenerateResume.delete(item)

    virtualResume = []
    count = 0
    for id in authorIds:
        pages = math.ceil(int(workCounts[count]) / 200)
        count += 1

        for page in range(pages):
            query = requests.get(
                "https://api.openalex.org/works?filter=author.id:" + id + "&page=" + str(page + 1) + "&per-page=200"
            )  # 200 per page
            results = json.loads(query.text)
            worksById = results["results"]

            for work in worksById:
                authorship = work["authorships"]
                authors = []
                for author in authorship:
                    authors.append(author["author"])
                cleanTitleCharacters = work["title"]
                cleanTitleCharacters = str(cleanTitleCharacters).replace("\n", "").replace("||", "--")
                date = work["publication_year"]
                type = work["type"]
                line = (
                    cleanTitleCharacters
                    + "||"
                    + str(date)
                    + "||"
                    + str(type)
                    + "||"
                    + str(authors)
                )
                # Remove duplication here
                flag = True
                for content in virtualResume:
                    if cleanTitleCharacters in content:
                        print(cleanTitleCharacters)
                        flag = False
                        break
                if flag:
                    virtualResume.append(line)

    file.writelines(str(authorIds) + "\n")

    for line in virtualResume:
        file.writelines(line + "\n")
    messagebox.showinfo("Resume Ready", "Resume of " + fileFirstName + "-" + fileLastName + " is ready.")
    file.close()

def loadTreeView():
    authorList = searchAuthorIds(firstNameEntry.get(), lastNameEntry.get())

    for item in treeSelectName.get_children():
        treeSelectName.delete(item)

    for author in authorList:
        treeSelectName.insert(
            "", tk.END, text=author["display_name"], values=(str(author["id"])[21:], author["works_count"])
        )

def on_doubleClick_treeSelectName(event):
    line = treeSelectName.focus()
    authorName = treeSelectName.item(line)["text"]
    authorId = treeSelectName.item(line)["values"][0]
    authorWorks = treeSelectName.item(line)["values"][1]

    treeGenerateResume.insert("", tk.END, text=authorName, values=(authorId, authorWorks))

    treeSelectName.delete(line)

def on_rightClick_treeSelectName(event):
    # absoluteCurrentDirectory = os.path.abspath("./")
    absoluteCurrentDirectory = os.path.abspath("../")
    directory = absoluteCurrentDirectory + "//evaluators//"
    file = open(directory + "temp.txt", "w", encoding="utf8")

    resumeOnlyTitle = []

    workCounts = []
    virtualResume = []
    item = treeSelectName.focus()
    id = treeSelectName.item(item)["values"][0]
    workCounts.append(treeSelectName.item(item)["values"][1])

    query = requests.get(
        "https://api.openalex.org/works?filter=author.id:" + id + "&per-page=200"
    )  # default is 20 per page
    results = json.loads(query.text)
    worksById = results["results"]

    for work in worksById:
        authorship = work["authorships"]
        authors = []
        for author in authorship:
            authors.append(author["author"])
        cleanTitleCharacters = work["title"]
        cleanTitleCharacters = str(cleanTitleCharacters).replace("\n", "").replace("||", "--")
        date = work["publication_year"]
        type = work["type"]
        resumeOnlyTitle.append(cleanTitleCharacters)
        line = (
            cleanTitleCharacters
            + "||"
            + str(date)
            + "||"
            + str(type)
            + "||"
            + str(authors)
        )
        # Remove duplication here
        flag = True
        for content in virtualResume:
            if cleanTitleCharacters in content:
                print(cleanTitleCharacters)
                flag = False
                break
        if flag:
            virtualResume.append(line)

    file.writelines(str(id) + "\n")


    for line in virtualResume:
        file.writelines(line + "\n")

    file.close()
    os.startfile(directory + "temp.txt")
    time.sleep(1)
    os.remove(directory + "temp.txt")

def on_doubleClick_treeGenerateResume(event):
    line = treeGenerateResume.focus()
    authorName = treeGenerateResume.item(line)["text"]
    authorId = treeGenerateResume.item(line)["values"][0]
    authorWorks = treeGenerateResume.item(line)["values"][1]

    treeSelectName.insert("", tk.END, text=authorName, values=(authorId, authorWorks))
    treeGenerateResume.delete(line)

window = tk.Tk()
window.title("Search OpenAlex Evaluators")
# window.state('zoomed')

firstName_label = tk.Label(window, text="Enter the suggestion's first name:")
firstName_label.grid(row=0, column=0)
firstNameEntry = tk.Entry(window)
firstNameEntry.grid(row=0, column=1)

lastName_label = tk.Label(window, text="Enter the suggestion's last name:")
lastName_label.grid(row=1, column=0)
lastNameEntry = tk.Entry(window)
lastNameEntry.grid(row=1, column=1)

searchButton = tk.Button(window, text="Search", command=loadTreeView)
searchButton.grid(row=2, column=0, columnspan=2)

response_label = tk.Label(window, text="Response to the query:")
response_label.grid(row=3, column=0)

treeSelectName = ttk.Treeview(window)
vsb = ttk.Scrollbar(window, orient="vertical", command=treeSelectName.yview)
treeSelectName.configure(yscrollcommand=vsb.set)
vsb.grid(row=4, column=3, rowspan=2)

treeSelectName["columns"] = ("id", "works")
treeSelectName.column("#0", width=250)
treeSelectName.column("id", width=100)
treeSelectName.column("works", width=100)
treeSelectName.heading("#0", text="Name")
treeSelectName.heading("id", text="ID")
treeSelectName.heading("works", text="Works")
treeSelectName.bind("<Double-1>", on_doubleClick_treeSelectName)
treeSelectName.bind("<Button-3>", on_rightClick_treeSelectName)  # Consult the record's resume to validate
treeSelectName.grid(row=4, column=0, columnspan=2)

selections_label = tk.Label(window, text="Selections to add to the resume:")
selections_label.grid(row=5, column=0)

treeGenerateResume = ttk.Treeview(window)
vsb = ttk.Scrollbar(window, orient="vertical", command=treeGenerateResume.yview)
treeGenerateResume.configure(yscrollcommand=vsb.set)
vsb.grid(row=6, column=3, rowspan=2)

treeGenerateResume["columns"] = ("id", "works")
treeGenerateResume.column("#0", width=250)
treeGenerateResume.column("id", width=100)
treeGenerateResume.column("works", width=100)
treeGenerateResume.heading("#0", text="Name")
treeGenerateResume.heading("id", text="ID")
treeGenerateResume.heading("works", text="Works")
treeGenerateResume.bind("<Double-1>", on_doubleClick_treeGenerateResume)

treeGenerateResume.grid(row=6, column=0, columnspan=2)

buildResumeButton = tk.Button(window, text="Generate virtual resume", command=buildResume)
buildResumeButton.grid(row=7, column=0, columnspan=2)

window.mainloop()