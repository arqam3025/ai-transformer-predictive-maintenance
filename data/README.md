\# Data



This directory contains data used by the AI Transformer Predictive Maintenance project.



\## Reference DGA Dataset



The processed reference dataset contains dissolved gas analysis (DGA)

measurements from 25 power transformers.



\### Source



\*\*Dataset:\*\* DGA results for 25 transformers  

\*\*Authors:\*\* Oleg Shutenko and Oleksii Kulyk  

\*\*Repository:\*\* Mendeley Data  

\*\*DOI:\*\* 10.17632/x22y4s9fks.1  

\*\*License:\*\* CC BY 4.0



Source page:

https://data.mendeley.com/datasets/x22y4s9fks/1



\## Measurements



The dataset contains concentrations for five dissolved gases:



\- H2 — Hydrogen

\- CH4 — Methane

\- C2H6 — Ethane

\- C2H4 — Ethylene

\- C2H2 — Acetylene



The original fault descriptions are retained in the processed dataset.



\## Processing



The original source document is stored locally under:



`data/raw/`



Raw PDF files are intentionally excluded from version control.



The machine-readable dataset is generated using:



`scripts/build\_validation\_dataset.py`



Output:



`data/processed/dga\_25\_transformers.csv`



The processing pipeline validates:



\- 25 transformer records are present

\- transformer identifiers are unique

\- all five DGA measurements are available

\- gas concentrations are non-negative

\- original fault descriptions are retained

\- combined diagnoses are not silently converted into single fault classes



\## Label Handling



Some source records contain explicit diagnostic labels such as:



`PD`, `T1`, `T2`, `T3`, and `D2`.



Other records contain combined diagnoses such as:



`T3+D2` and `T2+D1`.



These combined diagnoses are retained rather than forcing them into a

single class.



Descriptive diagnoses without an explicit single diagnostic class are

represented conservatively using labels such as:



`OTHER\_DISCHARGE` and `MIXED\_UNSPECIFIED`.



These labels are project-level data organisation labels and should not be

interpreted as IEC or IEEE diagnostic classifications.



\## Intended Use



This 25-transformer dataset is used as a small real-world engineering

reference and validation dataset.



Because the dataset contains only 25 observations, it is \*\*not used by

itself to claim production-grade machine-learning performance\*\*.



A larger, properly sourced dataset will be used for model training and

evaluation.



\## Attribution



The original measurements and diagnostic information belong to the

dataset authors.



The data-processing pipeline, feature engineering, validation tests,

machine-learning implementation, predictive-maintenance methodology and

software developed in this repository form part of the independent

AI Transformer Predictive Maintenance project by Raja Arqam Abdullah.

