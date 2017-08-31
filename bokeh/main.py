"""
Notional Customer 360 Dashboard.
"""

import pandas as pd
import logging
from os import listdir
from os.path import join, isfile
from bokeh.layouts import widgetbox, layout
from bokeh.models import ColumnDataSource, Div, Select, RadioButtonGroup
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, TextInput
import pyodbc
from bokeh.models import HoverTool
from datetime import datetime
import sys
import pydevd

# pydevd.settrace('10.1.2.100', port=5230, stdoutToServer=True, stderrToServer=True)

########################################################
# Define Div sections
########################################################

headliner = Div(text="""<div class="content"> <h1>Customer 360 powered by MapR&trade;</h1> <p>The <a 
href="https://mapr.com/products/mapr-converged-data-platform/">MapR Converged Data Platform</a> is uniquely suited to 
run Customer 360 applications. Operational and analytical workloads can operate together on the same cluster used for 
cloud scale storage, schema-free data integration, real-time streaming, and machine learning. Those capabilities 
enable applications to use more information in more ways than has ever been possible before. The application shown 
below uses those capabiltiies to help customer support representatives quickly determine customer personality, 
propensity to buy, and likelihood to churn. Check out the <a 
href="https://mapr.com/solutions/quickstart/customer-360-knowing-your-customer-is-step-one/">Customer 360 Quick Start 
Solution</a> to learn more about MapR's solutions for Customer 360 applications.</p> <div 
id="customer360_hype_container" style="margin:auto;position:relative;width:900px;height:400px;overflow:hidden;"> 
<script type="text/javascript" charset="utf-8" 
src="bokeh/static/js/customer360.hyperesources/customer360_hype_generated_script.js?34086"></script> </div> </div>""")
intro = Div(text="""<div class="content"><hr><h1>Customer Intelligence Portal for ACME Bank </h1></div>""")
customer_directory_title = Div(text="""<h3>Customer Directory:</h3>""")
ML_column_title = Div(text="""<h3>Machine Learning:</h3>""")
Persona_column_title = Div(text="""<h3>Selected Customer:</h3>""")

headshot = Div(text='<img src="bokeh/static/face_images/84b.jpg" alt="face" width="150">')
selected_name = Div(text='<p><strong>Name:</strong> Eva Peterson</p>')
needs = Div(text='<p>Needs:</p><ul>'
                 '<li>Home, car, and property insurance</li>'
                 '<li>To save for retirement</li>'
                 '<li>To exchange foreign currencies</li>'
                 '</ul>')
has = Div(text='<p>Has:</p><ul>'
               '<li>Savings Account</li>'
               '<li>Credit Card</li>'
               '<li>Roth IRA</li></ul>')
newline = Div(text="""<br>""")
query_performance = Div(text="""<div class="content"><br></div>""")


sys.setrecursionlimit(10000)

logger = logging.getLogger('bokeh')
logger.setLevel(logging.DEBUG)

conn = pyodbc.connect("DSN=drill64", autocommit=True)
# Specify unicode options only for MapR 5.2:
# conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
# conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)

cursor = conn.cursor()

sql = "SELECT _id, name, address, email, phone_number, latitude, longitude, first_visit, churn_risk, sentiment " \
      "FROM `dfs.default`.`./tmp/crm_data` limit 10000"
logger.debug("executing SQL: " + sql)
customer_directory_df = pd.read_sql(sql, conn)
logger.debug("records returned: " + str(len(customer_directory_df.index)))
query_performance.text = "<div class=\"small\">" + str(len(customer_directory_df.index)) + " rows selected</div>"

text_input = TextInput(title="Filter by Name:", width=180)
sort_options = ['name', 'phone_number', 'email', 'first_visit']
sortby = Select(title="Order by:",
                width=180,
                value="name",
                options=sort_options)
controls = [text_input, sortby]
for control in controls:
    control.on_change('value', lambda attr, old, new: customer_directory_filter())


def customer_directory_filter():
    # sorting by date requires converting the character string in first_visit
    if (sortby.value == 'first_visit'):
        sql = "SELECT _id, name, address, email, phone_number, latitude, longitude, first_visit, TO_DATE(`first_visit`, 'MM/dd/yyyy') AS first_visit_date_type, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` where name like '%" + text_input.value.strip() + "%' order by first_visit_date_type limit 10000"
    else:
        sql = "SELECT _id, name, address, email, phone_number, latitude, longitude, first_visit, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` where name like '%" + text_input.value.strip() + "%' order by " + sortby.value + " limit 10000"
    logger.debug("executing SQL: " + sql)
    global customer_directory_df, headshots, customer_directory_source
    customer_directory_df = pd.read_sql(sql, conn)
    logger.debug("records returned: " + str(len(customer_directory_df.index)))
    query_performance.text = "<div class=\"small\">" + str(len(customer_directory_df.index)) + " rows selected</div>"

    # Add headshot to each row of customer_directory_df
    # Load face image files for each customer
    headshots_path = "/home/mapr/customer360/bokeh/static/face_images/"
    headshots = [f for f in listdir(headshots_path) if isfile(join(headshots_path, f))]
    # We have a dataset of face images, but we may not have enough face images for every customer
    # So we'll just use the same images for some customers.
    if len(headshots) > len(customer_directory_df):
        headshots = headshots[:len(customer_directory_df)]
    while len(customer_directory_df) > len(headshots):
        headshots.extend(headshots[:(len(customer_directory_df) - len(headshots))])
    customer_directory_df['headshot'] = headshots

    # Add tenure to each row of customer_directory_df
    customer_directory_df['tenure'] = customer_directory_df['first_visit'].apply(
        lambda x: (datetime.today() - datetime.strptime(x, '%m/%d/%Y')).days)

    customer_directory_source.data = {
        'name': customer_directory_df.name,
        'phone_number': customer_directory_df.phone_number,
        'email': customer_directory_df.email,
        'tenure': ((customer_directory_df.tenure / 365).astype(int)).astype(str) + 'yr'
    }
    add_glyphs()


# Add headshot to each row of customer_directory_df
# Load face image files for each customer
headshots_path = "/home/mapr/customer360/bokeh/static/face_images/"
headshots = [f for f in listdir(headshots_path) if isfile(join(headshots_path, f))]
# We have a dataset of face images, but we may not have enough face images for every customer
# So we'll just use the same images for some customers.
if len(headshots) > len(customer_directory_df):
    headshots = headshots[:len(customer_directory_df)]
while len(customer_directory_df) > len(headshots):
    headshots.extend(headshots[:(len(customer_directory_df) - len(headshots))])
customer_directory_df['headshot'] = headshots
customer_directory_df['headshot'][0] = "84b.jpg"
customer_directory_df['headshot'][1] = "1b.jpg"
customer_directory_df['headshot'][2] = "78b.jpg"
customer_directory_df['headshot'][3] = "79b.jpg"
customer_directory_df['headshot'][4] = "54b.jpg"
customer_directory_df['headshot'][5] = "28b.jpg"
customer_directory_df['headshot'][6] = "43b.jpg"
customer_directory_df['headshot'][7] = "30b.jpg"
customer_directory_df['headshot'][8] = "36b.jpg"
customer_directory_df['headshot'][9] = "26b.jpg"


# Add tenure to each row of customer_directory_df
customer_directory_df['tenure'] = customer_directory_df['first_visit'].apply(
    lambda x: (datetime.today() - datetime.strptime(x, '%m/%d/%Y')).days)

customer_directory_source = ColumnDataSource(data=dict())

pageviews_reset = True


def selection_update(new):
    # Processes selections in the customer name directory table
    logger.debug(new)
    logger.debug(new.keys)
    logger.debug(new['1d'])
    inds = np.array(new['1d'][u'indices'])
    selected_names = customer_directory_source.data['name'][inds]
    current = customer_directory_df[customer_directory_df['name'].isin(selected_names)]
    selected_name.text = '<p><strong>Name:</strong> ' + str(current['name'].iloc[0]) + '</p>'

    image_file = "bokeh/static/face_images/" + str(current.iloc[0].headshot)
    logger.debug("Selected Names:")
    logger.debug(selected_names)
    logger.debug("Cumulative Churn Risk:")
    logger.debug(pd.DataFrame(current['churn_risk']).mean())
    logger.debug("Headshot file:")
    logger.debug(image_file)

    headshot.text = str('<img src=') + image_file + str(' alt="face" width="150">')

    machine_learning_table.data = {
        'Characteristic': ['Churn Risk', 'Sentiment', 'Persona', 'Upsell', 'Lifetime Value'],
        'Prediction': [str(int(current['churn_risk'].iloc[0])) + '%',
                       current['sentiment'].iloc[0],
                       'Group ' + str(np.random.randint(low=1, high=4)),
                       'Auto Loan',
                       '$' + str(500 + np.random.randint(low=0, high=499)) + ',' + str(
                           100 + np.random.randint(low=0, high=899))]
    }

    # Generate a new spend rate curve
    import random
    purchases = txndf[txndf['Amount'] < 0]
    purchases = purchases.iloc[::-1]
    purchases['Day'] = purchases['Date'].apply(
        lambda x: (x - purchases['Date'].iloc[0]) / np.timedelta64(1, 'D') + random.randint(-5, 5)).astype(int)

    purchases['Amount'] = purchases['Amount'].apply(lambda x: x * random.uniform(.5, 2))

    # # ensure purchase amounts are non-negative
    # purchases_update=pd.DataFrame.from_dict(data=dict(day=purchases['Day'],
    #                                                    date=purchases['Date'],
    #                                                    purchase_amount=abs(purchases['Amount']),
    #                                                    accumulated_purchases=abs(purchases['Amount']).cumsum()))

    purchase_history.data = dict(day=purchases['Day'],
                                 date=purchases['Date'],
                                 purchase_amount=abs(purchases['Amount']),
                                 accumulated_purchases=abs(purchases['Amount']).cumsum())

    cleaned_purchases = pd.DataFrame.from_dict(purchase_history.data)

    x = cleaned_purchases.day
    y = cleaned_purchases.accumulated_purchases

    lr = LinearRegression()
    lr.fit(x[:, np.newaxis], y)

    y_lr = lr.predict(x[:, np.newaxis])

    x = [purchases['Date'].iloc[0] + np.timedelta64(x.min() - 7, 'D'),
         purchases['Date'].iloc[0] + np.timedelta64(x.max() + 7, 'D')]
    y = [y_lr.min(), y_lr.max()]
    # lines_source = ColumnDataSource(data=dict(x=x, y=y))
    lines_source.data = dict(x=x, y=y)

    gmapsource.data = dict(
        desc=current.name.tolist(),
        lat=current.latitude.tolist(),
        lon=current.longitude.tolist(),
        tenure=current.glyphsize.tolist(),
        size=[10] * current.size,
        color=current.glyphcolor.tolist(),
        sentiment=current.sentiment.tolist()
    )
    gmapplot.add_glyph(gmapsource, circle)

    global pageviews_reset
    pageviews_reset = True


def make_default_selection():
    # Processes selections in the customer name directory table
    current = customer_directory_df[customer_directory_df['name'] == ('Eva Peterson')]
    image_file = "bokeh/static/face_images/" + str(current.iloc[0].headshot)
    headshot.text = str('<img src=') + image_file + str(' alt="face" width="150">')

    machine_learning_table.data = {
        'Characteristic': ['Churn Risk', 'Sentiment', 'Persona', 'Upsell', 'Lifetime Value'],
        'Prediction': [str(int(current['churn_risk'].iloc[0] / 100) + 1) + '%',
                       current['sentiment'].iloc[0],
                       'Group ' + str(np.random.randint(low=1, high=4)),
                       'Auto Loan',
                       '$' + str(500 + np.random.randint(low=0, high=499)) + ',' + str(
                           100 + np.random.randint(low=0, high=899))]
    }
    gmapsource.data = dict(
        desc=customer_directory_df.name.head(1000).tolist(),
        lat=customer_directory_df.latitude.head(1000).tolist(),
        lon=customer_directory_df.longitude.head(1000).tolist(),
        tenure=customer_directory_df.glyphsize.head(1000).tolist(),
        size=customer_directory_df.glyphsize.head(1000).tolist(),
        color=customer_directory_df.glyphcolor.head(1000).tolist(),
        sentiment=customer_directory_df.sentiment.head(1000).tolist()
    )
    gmapplot.add_glyph(gmapsource, circle)


columns = [
    TableColumn(field="name", title="Name", width=120),
    TableColumn(field="phone_number", title="Phone", width=100),
    TableColumn(field="email", title="Email", width=150),
    TableColumn(field="tenure", title="Tenure", width=60, formatter=StringFormatter())
    # TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="0.000%")),
]

customer_directory_table = DataTable(source=customer_directory_source, columns=columns, row_headers=False,
                                     editable=True, width=280, height=300, fit_columns=False)

customer_directory_source.on_change('selected', lambda attr, old, new: selection_update(new))
customer_directory_source.data = {
    'name': customer_directory_df.name,
    'phone_number': customer_directory_df.phone_number,
    'email': customer_directory_df.email,
    'tenure': ((customer_directory_df.tenure / 365).astype(int)).astype(str) + 'yr'
}

churn_table_columns = [
    TableColumn(field="Characteristic", title="Characteristic"),
    TableColumn(field="Prediction", title="Prediction")
    # TableColumn(field="Prediction", title="Probability", formatter=NumberFormatter(format="0.000%"))
]

machine_learning_table = ColumnDataSource(data=dict())

ML_table = DataTable(source=machine_learning_table, columns=churn_table_columns, row_headers=False, editable=True,
                     width=280, height=160)

##############################################################################
# Create heatmap
##########################

from heatmap import create_heatmap

hm = create_heatmap(customer_directory_df)

##############################################################################
# define geo map
##################

from bokeh.models import (
    GMapPlot, GMapOptions, ColumnDataSource, Circle, LinearColorMapper, DataRange1d
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


# size=(datetime.today() - datetime.strptime('01/13/1999', '%m/%d/%Y')).days
def add_glyphs():
    global customer_directory_df
    max = customer_directory_df['tenure'].loc[customer_directory_df['tenure'].idxmax()]
    customer_directory_df['glyphsize'] = customer_directory_df['tenure'].apply(lambda x: (10 * x / max))
    customer_directory_df['glyphcolor'] = pd.Series(customer_directory_df.sentiment, dtype="category")
    customer_directory_df['glyphcolor'] = customer_directory_df['glyphcolor'].cat.rename_categories(
        ['red', 'grey', 'blue'])


add_glyphs()

# Data to be visualized (equal length arrays)
gmapsource = ColumnDataSource(
    data=dict(
        desc=customer_directory_df.name.tolist(),
        lat=customer_directory_df.latitude.tolist(),
        lon=customer_directory_df.longitude.tolist(),
        size=customer_directory_df.glyphsize.tolist(),
        tenure=customer_directory_df.glyphsize.tolist(),
        # size=housing.median_income.tolist(),
        color=customer_directory_df.glyphcolor.tolist(),
        sentiment=customer_directory_df.sentiment.tolist()
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
# Linear regression plot
################################

import pandas as pd
import numpy as np
from bokeh.plotting import *
from bokeh.models import ColumnDataSource, DatetimeTickFormatter, NumeralTickFormatter, PrintfTickFormatter
from math import pi
from bokeh.models.glyphs import Line

# TODO: put this in maprdb and query it with Drill
DATA_FILE = "/home/mapr/customer360/bokeh/datasets/credit_card_transactions.csv"
txndf = pd.read_csv(DATA_FILE, thousands=',')
txndf['Date'] = pd.to_datetime(txndf['Date'], format='%m/%d/%Y')

# ignore refunds and deposits to credit card account
purchases = txndf[txndf['Amount'] < 0]
# reverse the data so the accumulator starts from oldest date
purchases = purchases.iloc[::-1]
# # create a column to count days elapsed since first purchase

purchases['Day'] = purchases['Date'].apply(lambda x: (x - purchases['Date'].iloc[0]) / np.timedelta64(1, 'D')).astype(
    int)

# # ensure purchase amounts are non-negative
cleaned_purchases = pd.DataFrame.from_dict(data=dict(day=purchases['Day'],
                                                     date=purchases['Date'],
                                                     purchase_amount=abs(purchases['Amount']),
                                                     accumulated_purchases=abs(purchases['Amount']).cumsum()))
purchase_history = ColumnDataSource(data=cleaned_purchases)

from sklearn.linear_model import LinearRegression

x = cleaned_purchases.day
y = cleaned_purchases.accumulated_purchases

lr = LinearRegression()
lr.fit(x[:, np.newaxis], y)
y_lr = lr.predict(x[:, np.newaxis])
days_in_a_lifetime = [[600]]
lifetime_value = "{:,}".format(int(lr.predict(days_in_a_lifetime)[0]) * 10)

# create and render a scatter plot for accumulated purchases
plt = figure(width=280, height=200, x_axis_type="datetime", title='Spend Rate', toolbar_location=None)
plt.xaxis.formatter = DatetimeTickFormatter(months=["%b %Y"])
plt.yaxis.formatter = NumeralTickFormatter(format="‘$0,0’")
plt.circle('date', 'accumulated_purchases', source=purchase_history, size=2)
# plt.line([purchases['Date'].iloc[0] + np.timedelta64(x.min(), 'D'),
#           purchases['Date'].iloc[0] + np.timedelta64(x.max(), 'D')],
#          [y_lr.min(), y_lr.max()], color='red', line_width=2)
plt.xaxis.major_label_orientation = pi / 4

x = [purchases['Date'].iloc[0] + np.timedelta64(x.min() - 7, 'D'),
     purchases['Date'].iloc[0] + np.timedelta64(x.max() + 7, 'D')]
y = [y_lr.min(), y_lr.max()]
lines_source = ColumnDataSource(data=dict(x=x, y=y))
line = Line(x='x', y='y', line_color="red", line_width=2, line_alpha=.8)
plt.add_glyph(lines_source, line)

###############################################################################
# continuous animation graph for page views
#######################################
from numpy import asarray, cumprod, clip, ones, arange
from numpy.random import lognormal, rand, choice
from bokeh.layouts import column, gridplot
from bokeh.plotting import curdoc, figure
from bokeh.models import FuncTickFormatter
from bokeh.driving import count
from bokeh.models import LinearAxis, Range1d

averages = choice([2, 1], size=100, p=[.3, .7])
times = arange(0, 100, 1)

cont_source = ColumnDataSource(dict(
    time=[], average=[]
))

pageview_plt = figure(plot_height=150, x_axis_location=None, y_axis_location="left", title='Click Stream')
pageview_plt.x_range.follow = "end"
pageview_plt.yaxis.axis_label = "Pageviews"
pageview_plt.yaxis.ticker = [0.5, 1.0, 1.5, 2.0]
pageview_plt.extra_x_ranges = {"foo": Range1d(start=-60, end=0)}
pageview_plt.add_layout(LinearAxis(x_range_name="foo"), 'below')
pageview_plt.y_range = Range1d(0, 2, bounds='auto')
pageview_plt.xaxis[0].formatter = PrintfTickFormatter(format="%d sec")
pageview_plt.vbar(x='time', width=10, top='average', source=cont_source)
pageview_plt.xaxis.major_label_orientation = pi / 4
pageview_plt.xgrid.grid_line_color = None
pageview_plt.ygrid.grid_line_color = None


def _create_datapoints():
    average = choice([2, 1, -1], size=1, p=[.002, .008, .990])
    return average[0]


reset_t = -1
# @count() annotation increments t every time cont_update is called
t=0
def cont_update():
    # initialize or reset the pageviews graph
    global reset_t, pageviews_reset, t
    t=t+1
    # if reset is true, write a bunch of new values to the graph
    if pageviews_reset == True:
        pageviews_reset = False
        average = []
        for x in range(200):
            average.append(-1)
        for x in range(400):
            average.append(_create_datapoints())
        newdata = dict(
            time=[t+x for x in range(600)],
            average=average
        )
        t=t+600
        cont_source.stream(new_data=newdata, rollover=600)

    else:
        average = _create_datapoints()
        newdata = dict(
            time=[t],
            average=[average]
        )
        cont_source.stream(new_data=newdata, rollover=600)



##############################################################################
# layout the page elements
##########################

title = widgetbox(intro, width=700)
column1 = column(widgetbox(customer_directory_title, text_input, sortby, newline, customer_directory_table, query_performance, width=300))
column2 = column(widgetbox(ML_column_title, ML_table, width=300),
                 gridplot([[plt], [pageview_plt]], toolbar_location=None, plot_width=300))
column3 = widgetbox(Persona_column_title, headshot, selected_name, needs, has, width=300)
# drill_table_widget = widgetbox(drill_table)

curdoc().title = "Customer 360 Analytics"

headline_row = [headliner]
row1 = [title]
row2 = [column1, column2, column3]
row3 = [hm]
row4 = [gmapplot]
l1 = layout([row1, row2, row3, row4], sizing_mode='fixed')
l2 = layout([headline_row], sizing_mode='fixed')
curdoc().add_root(l1)
curdoc().add_root(l2)
# curdoc().add_root(gridplot([[p2]], toolbar_location="left", plot_width=500))
curdoc().add_periodic_callback(cont_update, 100)
curdoc().title = "Customer 360 Demo"

make_default_selection()
