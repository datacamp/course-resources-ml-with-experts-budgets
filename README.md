# course-resources-ml-with-experts-budgets
Further student resources for DrivenData's 'Machine Learning with the Experts: School Budgets' DataCamp course.

To see the model, take a look at the [notebook that builds the winning model](notebooks/1.0-full-model.ipynb).

To get the data, sign up for [the competition](https://www.drivendata.org/competitions/46/box-plots-for-education-reboot/) and use the data download link!

To run the notebook, first install the dependencies with:

    pip install -r requirements.txt

Then run:

    jupyter notebook notebooks/1.0-full-model.ipynb


Project Organization
------------

    ├── LICENSE
    ├── README.md   
    ├── data
    │   ├── TestSet.csv
    │   └── TrainingSet.csv
    ├── notebooks
    │   └── 1.0-full-model.ipynb
    ├── requirements.txt
    └── src
        ├── __init__.py
        ├── data
        │   └── multilabel.py
        ├── features
        │   └── SparseInteractions.py
        └── models
            └── metrics.py

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
