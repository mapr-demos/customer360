"""
Notional Customer 360 Dashboard.
"""

import pandas as pd
import numpy as np
import logging
from sklearn.datasets import make_classification, make_regression
from os import listdir
from os.path import dirname, join, isfile
from bokeh.layouts import row, widgetbox, column, gridplot, layout
from bokeh.models import ColumnDataSource, CustomJS, FixedTicker, Div, PreText
from bokeh.io import curdoc
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, NumberFormatter, AutocompleteInput
import pyodbc
from bokeh.models import HoverTool
from datetime import datetime

logger = logging.getLogger('bokeh')
logger.setLevel(logging.DEBUG)

conn = pyodbc.connect("DSN=drill64", autocommit=True)
conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)

cursor = conn.cursor()

sql="SELECT _id, name, address, phone_number, latitude, longitude, zip, first_visit, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` limit 10000 "
df=pd.read_sql(sql, conn)

# Load face image files for each customer
headshots_path="/home/mapr/customer360/bokeh/static/face_images/"
headshots = [f for f in listdir(headshots_path) if isfile(join(headshots_path, f))]
# We have a dataset of face images, but we may not have enough face images for every customer
# So we'll just use the same images for some customers.
if len(headshots) > len(df):
    headshots = headshots[:len(df)]
while len(df) > len(headshots):
    headshots.extend(headshots[:(len(df) - len(headshots))])
df['headshot'] = headshots
# print(df['headshot'][0])
# print(df['headshot'][len(df)-1])

source = ColumnDataSource(data=dict())


def selection_update(new):
    import random
    # Processes selections in the customer name directory table
    logger.debug(new)
    logger.debug(new.keys)
    logger.debug(new['1d'])
    inds = np.array(new['1d'][u'indices'])
    selected_names = source.data['name'][inds]
    current = df[df['name'].isin(selected_names)]
    image_file = "bokeh/static/face_images/"+str(current.iloc[0].headshot)

    logger.debug("Selected Names:")
    logger.debug(selected_names)
    logger.debug("Cumulative Churn Risk:")
    logger.debug(pd.DataFrame(current['churn_risk']).mean())
    logger.debug("Headshot file:")
    logger.debug(image_file)

    headshot.text = str('<img src=') + image_file + str(' alt="face" width="150">')

    machine_learning_table.data={
        'Characteristic': ['Churn Risk','Sentiment','Persona','Upsell','Lifetime Value'],
        'Prediction': [str(current['churn_risk'].iloc[0]/100)+'%',
                       current['sentiment'].iloc[0],
                       'Group 1',
                       'Auto Loan',
                       '$524,570']
    }

    # Generate a new spend rate curve
    purchases=txndf[txndf['Amount']<0]
    purchases=purchases.iloc[::-1]
    purchases['Day'] = purchases['Date'].apply(lambda x: (x - purchases['Date'].iloc[0]) / np.timedelta64(1, 'D') + random.randint(-5,5)).astype(int)

    purchases['Amount'] = purchases['Amount'].apply(lambda x: x*random.uniform(.5,2))

    # # ensure purchase amounts are non-negative
    # purchases_update=pd.DataFrame.from_dict(data=dict(day=purchases['Day'],
    #                                                    date=purchases['Date'],
    #                                                    purchase_amount=abs(purchases['Amount']),
    #                                                    accumulated_purchases=abs(purchases['Amount']).cumsum()))

    purchase_history.data = dict(day=purchases['Day'],
                                 date=purchases['Date'],
                                 purchase_amount=abs(purchases['Amount']),
                                 accumulated_purchases=abs(purchases['Amount']).cumsum())

    cleaned_purchases=pd.DataFrame.from_dict(purchase_history.data)

    x=cleaned_purchases.day
    y=cleaned_purchases.accumulated_purchases

    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)

    y_lr = lr.predict(x[:, np.newaxis])

    x = [purchases['Date'].iloc[0] + np.timedelta64(x.min()-7, 'D'),
         purchases['Date'].iloc[0] + np.timedelta64(x.max()+7, 'D')]
    y=[y_lr.min(), y_lr.max()]
    # lines_source = ColumnDataSource(data=dict(x=x, y=y))
    lines_source.data = dict(x=x, y=y)

    gmapsource.data=dict(
        desc=current.name.tolist(),
        lat=current.latitude.tolist(),
        lon=current.longitude.tolist(),
        tenure=current.glyphsize.tolist(),
        size=[10] * current.size,
        color=current.glyphcolor.tolist(),
        sentiment=current.sentiment.tolist()
    )
    gmapplot.add_glyph(gmapsource, circle)

def make_default_selection():
    # Processes selections in the customer name directory table
    current = df[df['name'] == ('Erika Gallardo')]
    image_file = "bokeh/static/face_images/"+str(current.iloc[0].headshot)
    headshot.text = str('<img src=') + image_file + str(' alt="face" width="150">')

    machine_learning_table.data={
        'Characteristic': ['Churn Risk','Sentiment','Persona','Upsell','Lifetime Value'],
        'Prediction': [str(current['churn_risk'].iloc[0]/100)+'%',
                       current['sentiment'].iloc[0],
                       'Group 1',
                       'Auto Loan',
                       '$524,570']
    }
    gmapsource.data=dict(
        desc=df.name.head(1000).tolist(),
        lat=df.latitude.head(1000).tolist(),
        lon=df.longitude.head(1000).tolist(),
        tenure=df.glyphsize.head(1000).tolist(),
        size=df.glyphsize.head(1000).tolist(),
        color=df.glyphcolor.head(1000).tolist(),
        sentiment=df.sentiment.head(1000).tolist()
    )
    gmapplot.add_glyph(gmapsource, circle)

def update():
    # Process changes to the slider
    # logger.debug("Slider value:")
    # logger.debug(slider.value)
    current = df[(df['name'].str.contains(autocomplete.value))].head(slider.value)

    source.data = {
        'name'           : current.name,
        'phone_number'   : current.phone_number,
        'address '       : current.address
    }

    machine_learning_table.data={
        'Characteristic': ['Upsell','Churn Risk','Sentiment','Persona','Lifetime Value'],
        'Prediction': ['Auto Loan',
                       str(current['churn_risk'].iloc[0]/100)+'%',
                       current['sentiment'].iloc[0],
                       'Group 1',
                       '$524,570']
    }

    gmapsource.data=dict(
        desc=current.name.tolist(),
        lat=current.latitude.tolist(),
        lon=current.longitude.tolist(),
        tenure=current.glyphsize.tolist(),
        size=[10] * current.size,
        color=current.glyphcolor.tolist(),
        sentiment=current.sentiment.tolist()
    )
    gmapplot.add_glyph(gmapsource, circle)


slider = Slider(title="Filter by Top Names", start=1, end=5000, value=1000, step=100)
slider.on_change('value', lambda attr, old, new: update())

autocomplete = AutocompleteInput(title="Filter by Name", completions=df['name'].tolist())
autocomplete.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download.js")).read())

columns = [
    TableColumn(field="name", title="Name"),
    TableColumn(field="phone_number", title="Phone")
    #TableColumn(field="address", title="Address")
    #TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="0.000%")),
]

customer_directory_table = DataTable(source=source, columns=columns, row_headers=False, editable=True, width=280)
source.on_change('selected', lambda attr, old, new: selection_update(new))
source.data = {
    'name'           : df.name,
    'phone_number'   : df.phone_number,
    'address '       : df.address
}

churn_table_columns = [
    TableColumn(field="Characteristic", title="Characteristic"),
    TableColumn(field="Prediction", title="Prediction")
    # TableColumn(field="Prediction", title="Probability", formatter=NumberFormatter(format="0.000%"))
]

machine_learning_table = ColumnDataSource(data=dict())

ML_table = DataTable(source=machine_learning_table, columns=churn_table_columns, row_headers=False, editable=True, width=280, height=200)

##############################################################################
# Create heatmap
##########################

from heatmap import create_heatmap
hm = create_heatmap(df)


##############################################################################
# define geo map
##################

from bokeh.models import (
    GMapPlot, GMapOptions, ColumnDataSource, Circle, LinearColorMapper, BasicTicker, ColorBar,
    DataRange1d
)

# map_options = GMapOptions(lat=37.6, lng=-119.5, map_type="roadmap", zoom=6)
map_options = GMapOptions(lat=32, lng=-96, map_type="roadmap", zoom=4)

# For GMaps to function, Google requires you obtain and enable an API key:
#
#     https://developers.google.com/maps/documentation/javascript/get-api-key
#
# Replace the value below with your personal API key:
gmapplot = GMapPlot(
    api_key="AIzaSyBYrbp34OohAHsX1cub8ZeHlMEFajv15fY",
    x_range=DataRange1d(), y_range=DataRange1d(), map_options=map_options, width=900,
    toolbar_location="right",
)

gmapplot.title.text = "Customer Location, Tenure, and Sentiment"

#current['first_visit'] = current['first_visit'].apply(lambda x: 50000)
#size=(datetime.today() - datetime.strptime('01/13/1999', '%m/%d/%Y')).days
df['tenure'] = df['first_visit'].apply(lambda x: (datetime.today() - datetime.strptime(x, '%m/%d/%Y')).days)
max = df['tenure'].loc[df['tenure'].idxmax()]
df['glyphsize'] = df['tenure'].apply(lambda x: (10 * x/max))
df['glyphcolor'] = pd.Series(df.sentiment, dtype="category")
df['glyphcolor'] = df['glyphcolor'].cat.rename_categories(['red','grey','blue'])

# Data to be visualized (equal length arrays)
gmapsource = ColumnDataSource(
    data=dict(
        desc=df.name.tolist(),
        lat=df.latitude.tolist(),
        lon=df.longitude.tolist(),
        size=df.glyphsize.tolist(),
        tenure=df.glyphsize.tolist(),
        # size=housing.median_income.tolist(),
        color=df.glyphcolor.tolist(),
        sentiment=df.sentiment.tolist()
    )
)

color_mapper = LinearColorMapper(palette="Viridis5")

# Circle glyph with relevant properties for plotting
circle = Circle(x="lon", y="lat", size="size", fill_color="color",
                fill_alpha=0.5, line_color=None)
gmapplot.add_glyph(gmapsource, circle)

# Information to be displayed when hovering over glyphs
gmaphover = HoverTool(
    tooltips=[
        ("Name", "@desc"),
        ("Tenure", "@tenure years"),
        ("Sentiment", "@sentiment")
    ]
)
gmapplot.add_tools(gmaphover)


##############################################################################
# Drill queries
# NOTE!  This won't work with anaconda panda. It throws a symbol error.
##############################

# Initialize the connection
# The DSN was defined with the iODBC Administrator app for Mac.
conn = pyodbc.connect("DSN=drill64", autocommit=True)
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le')
cursor = conn.cursor()

# Setup a SQL query to select data from a csv file.
# The csv2 filename extension tells Drill to extract
# column names from the first row.
# sql = "SELECT * FROM `dfs.tmp`.`./companylist.csv2` limit 3"
# Execute the SQL query
# print(pandas.read_sql(sql, conn))
# Here's how to select data from MySQL
# s = "select * from ianmysql.mysql.`user`"
# pandas.read_sql(s, conn)

# Here's an example of a SQL JOIN the combines a JSON file with a MySQL table.
sql = "SELECT tbl1.name, tbl2.address FROM `dfs.tmp`.`./names.json` as tbl1 \
     JOIN `dfs.tmp`.`./addressunitedstates.json` as tbl2 ON tbl1.id=tbl2.id"


drill_data = ColumnDataSource(data=pd.read_sql(sql, conn))
drill_columns = [
    TableColumn(field="name", title="Name"),
    TableColumn(field="phone_number", title="Phone"),
    TableColumn(field="first_visit", title="Tenure")
]
drill_table = DataTable(source=drill_data, columns=drill_columns, row_headers=False, width=600)

##############################################################################
# Linear regression plot
################################

import pandas as pd
import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, NumeralTickFormatter
from math import pi
from bokeh.models.glyphs import Line

DATA_FILE = "/home/mapr/credit_card_transactions.csv"
txndf=pd.read_csv(DATA_FILE, thousands=',')
txndf['Date'] = pd.to_datetime(txndf['Date'], format='%m/%d/%Y')

# ignore refunds and deposits to credit card account
purchases=txndf[txndf['Amount']<0]
# reverse the data so the accumulator starts from oldest date
purchases=purchases.iloc[::-1]
# # create a column to count days elapsed since first purchase

purchases['Day'] = purchases['Date'].apply(lambda x: (x - purchases['Date'].iloc[0]) / np.timedelta64(1, 'D')).astype(int)

# # ensure purchase amounts are non-negative
cleaned_purchases=pd.DataFrame.from_dict(data=dict(day=purchases['Day'],
                                                   date=purchases['Date'],
                                                   purchase_amount=abs(purchases['Amount']),
                                                   accumulated_purchases=abs(purchases['Amount']).cumsum()))
purchase_history = ColumnDataSource(data=cleaned_purchases)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
x=cleaned_purchases.day
y=cleaned_purchases.accumulated_purchases

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)
y_lr = lr.predict(x[:, np.newaxis])
days_in_a_lifetime = [[600]]
lifetime_value = "{:,}".format(int(lr.predict(days_in_a_lifetime)[0])*10)

# create and render a scatter plot for accumulated purchases
plt = figure(width=280, height=200, x_axis_type="datetime", title='Spend Rate', toolbar_location=None)
plt.xaxis.formatter = DatetimeTickFormatter(months=["%b %Y"])
plt.yaxis.formatter = NumeralTickFormatter(format="‘$0,0’")
plt.circle('date', 'accumulated_purchases', source=purchase_history, size=2)
# plt.line([purchases['Date'].iloc[0] + np.timedelta64(x.min(), 'D'),
#           purchases['Date'].iloc[0] + np.timedelta64(x.max(), 'D')],
#          [y_lr.min(), y_lr.max()], color='red', line_width=2)
plt.xaxis.major_label_orientation = pi/4

x = [purchases['Date'].iloc[0] + np.timedelta64(x.min()-7, 'D'),
      purchases['Date'].iloc[0] + np.timedelta64(x.max()+7, 'D')]
y=[y_lr.min(), y_lr.max()]
lines_source = ColumnDataSource(data=dict(x=x, y=y))
line = Line(x='x', y='y', line_color="red", line_width=2, line_alpha=.8)
plt.add_glyph(lines_source, line)


##############################################################################
# layout the page elements
##########################

headliner = Div(text="""<div class="content">
      <h1>Customer 360 powered by MapR&trade;</h1>
      <p>The <a href="https://mapr.com/products/mapr-converged-data-platform/">MapR Converged Data Platform</a> is uniquely suited to run Customer 360 applications. Operational and analytical workloads can co-operate on Hadoop data lakes and real-time streams containing large amounts of unstructured and semi-structured data. MapR provides cloud scale storage, schema-free data integration, and distributed event streaming so customer intelligence can be achieved through machine learning on datasets that relate to customer personality, sentiment, propensity to buy, and likelihood to churn. Check out the <a href="https://mapr.com/solutions/quickstart/customer-360-knowing-your-customer-is-step-one/">Customer 360 Quick Start Solution</a> to learn more about MapR's products and solutions for Customer 360 applications.</p>
      <div id="customer360_hype_container" style="margin:auto;position:relative;width:900px;height:400px;overflow:hidden;">
		<script type="text/javascript" charset="utf-8" src="bokeh/static/js/customer360.hyperesources/customer360_hype_generated_script.js?34086"></script>
	</div>
  </div>""")
intro = Div(text="""<div class="content"><hr><h1>Customer Intelligence for ACME Federal Savings Bank</h1></div>""")
customer_directory_title = Div(text="""<h3>Customer Directory:</h3>""")
ML_column_title = Div(text="""<h3>Machine Learning:</h3>""")
Persona_column_title = Div(text="""<h3>Survey Feedback:</h3>""")

headshot = Div(text='<img src="bokeh/static/face_images/10b.jpg" alt="face" width="150">')
needs = Div(text='<p>Needs:</p><ul>'
                 '<li>Home, car, and property insurance</li>'
                 '<li>To save for retirement</li>'
                 '<li>To exchange foreign currencies</li>'
                 '</ul>')
has = Div(text='<p>Has:</p><ul>'
               '<li>Savings Account</li>'
               '<li>Credit Card</li>'
               '<li>Roth IRA</li></ul>')

title = widgetbox(intro, width=700)
column1 = widgetbox(customer_directory_title, customer_directory_table, width=300)
column2 = column(widgetbox(ML_column_title, ML_table, width=300), plt)
column3 = widgetbox(Persona_column_title, headshot, needs, has, width=300)
# drill_table_widget = widgetbox(drill_table)

curdoc().title = "Customer 360 Analytics"

headline_row = [headliner]
row1 = [title]
row2 = [column1, column2, column3]
row3 = [hm]
row4 = [gmapplot]
# row5 = [drill_table_widget]
l = layout([headline_row], sizing_mode='fixed')
l2 = layout([row1, row2, row3, row4], sizing_mode='fixed')
curdoc().add_root(l)
curdoc().add_root(l2)

make_default_selection()
