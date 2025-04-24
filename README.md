# RL-Project-Hospital
(Optional) run the data_extraction.py file in the Code folder.
This step is optional as the data has already been web scrapped from the Project Hospital Wiki, and converted to CSV files in the Data folder.

The Agent.py file holds the code for the agent.

The Environment.py file holds the code for the environment.

Each .ipynb file (Emergency_Dep_main, Gen_Surgery_Dep_main, Neuro_Dep_main, Combined_main) hold all results and figures for the project (Figures can also be found in the Figures folder).
These files can be independently ran again to replicate results.
Note that Jupyter Notebook was used to run these .ipynb files with this exact file configuration to import the CSV files, Agent.py file, and Environment.py file.
If these files are executed in a different way or with a different file structure, changes to the import and read_csv function might have to be made.