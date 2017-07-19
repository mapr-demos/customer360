##########################
# Create heatmap
##########################

import pandas as pd
from sklearn.datasets import make_classification
from bokeh.models import ColumnDataSource, FixedTicker
from bokeh.charts import HeatMap
from bokeh.models.widgets import  DataTable, TableColumn
from bokeh.models import HoverTool

def create_heatmap(df):

    products=['Debit Card',
              'Personal Credit Card',
              'Business Credit Card',
              'Home Mortgage Loan',
              'Auto Loan',
              'Brokerage Account',
              'Roth IRA',
              '401k',
              'Home Insurance',
              'Automobile Insurance',
              'Medical Insurance',
              'Life Insurance',
              'Cell Phone',
              'Landline'
              ]

    def rename_columns(df):
        df = df.copy()
        df.columns = [products[i] for i in df.columns]
        return df

    # create an artificial dataset with 3 clusters
    X, Y = make_classification(n_samples=100, n_classes=4, n_features=12, n_redundant=0, n_informative=12,
                               scale=1000, n_clusters_per_class=1)
    df2 = pd.DataFrame(X)
    # ensure all values are positive (this is needed for our customer 360 use-case)
    df2 = df2.abs()
    # rename X columns
    df2 = rename_columns(df2)
    # and add the Y
    df2['y'] = Y
    #df
    # split df into cluster groups
    grouped = df2.groupby(['y'], sort=True)

    # compute sums for every column in every group
    sums = grouped.sum()

    score = []
    for x in sums.apply(tuple):
        score.extend(x)

    data=dict(
        persona=list(sums.index) * len(sums.columns),
        product=[item for item in list(sums.columns) for i in range(len(sums.index))],
        score=score
    )

    hm = HeatMap(data, x='product', y='persona', values='score', title='Customer Profiles', xlabel='Product', ylabel='Persona', legend=False, stat=None,  tools=["save"], height=400, width=900, toolbar_location=None)
    hm.yaxis.ticker=FixedTicker(ticks=[0,1,2,3])
    hm_source = ColumnDataSource(data=sums)

    hm_data_table = DataTable(
        source=hm_source,
        columns=[TableColumn(field=c, title=c) for c in sums.columns], width=900, height=150)

    return hm