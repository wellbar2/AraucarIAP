import tkinter as tk
from tkinter import messagebox
import subprocess
import platform
import os

def execute():
    startYear_param = startYear.get()
    endYear_param = endYear.get()
    affinityWeight_param = affinityWeight.get()
    coiWeight_param = coiWeight.get()
    evaluatorsPerProject_param = evaluatorsPerProject.get()
    projectsPerEvaluator_param = projectsPerEvaluator.get()

    operatingSystem = platform.system()

    if operatingSystem == "Windows":
        python = os.path.join('python')
    elif operatingSystem == "Linux":
        python = os.path.join('python')
    else:
        python_venv = None
        messagebox.showerror("Error", f"Operating system '{operatingSystem}' not supported.")

    command = [
        python, 'createNetwork.py',
        startYear_param,
        endYear_param,
        affinityWeight_param,
        coiWeight_param,
        evaluatorsPerProject_param,
        projectsPerEvaluator_param
    ]

    subprocess.run(command)

window = tk.Tk()
window.title("Create NETWORKS")

startYear_label = tk.Label(window, text="Enter the starting year:")
startYear_label.grid(row=0, column=0)
startYear = tk.Entry(window)
startYear.grid(row=0, column=1)

endYear_label = tk.Label(window, text="Enter the ending year:")
endYear_label.grid(row=1, column=0)
endYear = tk.Entry(window)
endYear.grid(row=1, column=1)

empty_label = tk.Label(window, text="    ")
empty_label.grid(row=2, column=0)

affinityWeight_label = tk.Label(window, text="Value of affinity weight X:")
affinityWeight_label.grid(row=3, column=0)
affinityWeight = tk.Entry(window)
affinityWeight.insert(0, "1")
affinityWeight.grid(row=3, column=1)

coiWeight_label = tk.Label(window, text="Value of CoI weight Y:")
coiWeight_label.grid(row=3, column=3)
coiWeight = tk.Entry(window)
coiWeight.insert(0, "1")
coiWeight.grid(row=3, column=4)

empty2_label = tk.Label(window, text="    ")
empty2_label.grid(row=4, column=0)

evaluatorsPerProject_label = tk.Label(window, text="Evaluators per project:")
evaluatorsPerProject_label.grid(row=5, column=0)
evaluatorsPerProject = tk.Entry(window)
evaluatorsPerProject.insert(0, "1")
evaluatorsPerProject.grid(row=5, column=1)

projectsPerEvaluator_label = tk.Label(window, text="Maximum projects per evaluator:")
projectsPerEvaluator_label.grid(row=5, column=3)
projectsPerEvaluator = tk.Entry(window)
projectsPerEvaluator.insert(0, "1")
projectsPerEvaluator.grid(row=5, column=4)

createButton = tk.Button(window, text="Create", command=execute)
createButton.grid(row=6, column=0, columnspan=2)

window.mainloop()