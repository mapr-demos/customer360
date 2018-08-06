"""
Notional Customer 360 Dashboard.
"""

import pandas as pd
import logging
from os import listdir
from os.path import join, isfile
from bokeh.layouts import widgetbox, layout
from bokeh.models import ColumnDataSource, Div, Select, RadioButtonGroup
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, TextInput, Button
import pyodbc
from bokeh.models import HoverTool
from datetime import datetime
import sys

# Uncomment to trace with IntelliJ / PyCharm debugger
# import pydevd
# pydevd.settrace('10.1.2.100', port=5230, stdoutToServer=True, stderrToServer=True)

########################################################
# Define Div sections
########################################################

headliner = Div(text="""<div class="content"><hr>
<div 
id="customer360_hype_container" style="margin:auto;position:relative;width:900px;height:400px;overflow:hidden;"> 
<script type="text/javascript" charset="utf-8" 
src="bokeh/static/js/customer360.hyperesources/customer360_hype_generated_script.js?34086"></script> </div>
<h1>Customer 360 powered by MapR&trade;</h1> <h3><p>The <a 
href="https://mapr.com/products/mapr-converged-data-platform/">MapR Converged Data Platform</a> is uniquely suited to 
run Customer 360 applications.</p> 
<ul>
<li>Cloud scale storage</li>
<li>Schema-free data integration</li>
<li>Access to real-time streaming data</li>
<li>Native support for Apache Spark</li>
</ul>
<p>Check out the <a 
href="https://mapr.com/solutions/quickstart/customer-360-knowing-your-customer-is-step-one/">Customer 360 Quick Start 
Solution</a> to learn more!</p></h3>  </div>""")
intro = Div(text="""<div class="content"><img src = "bokeh/static/images/topbanner.png" alt="Call Center Analytics for Retail Banking" width="760" height="96"></div>""")
customer_directory_title = Div(text="""<h3>Customer Directory:</h3>""")
ML_column_title = Div(text="""<h3>Next Best Action:</h3>""")
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
# Unicode options for Anaconda python
conn.setencoding("utf-8")
conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le')
conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le')
# Unicode options for python (not anaconda)
#conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
#conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)

cursor = conn.cursor()

sql = "SELECT _id, name, address, email, phone_number, latitude, longitude, first_visit, churn_risk, sentiment " \
      "FROM `dfs.default`.`./apps/crm` limit 10000"
logger.debug("executing SQL: " + sql)
customer_directory_df = pd.read_sql(sql, conn)
logger.debug("records returned: " + str(len(customer_directory_df.index)))
query_performance.text = "<div class=\"small\">" + str(len(customer_directory_df.index)) + " rows selected</div>"

text_input = TextInput(title="Search String:", value='')
filter_options = ['name', 'phone_number', 'email']
filterby = Select(title="Search Field:",
                width=100,
                value="name",
                options=filter_options)

sort_options = ['name', 'phone_number', 'email', 'first_visit']
sortby = Select(title="Order By:",
                width=100,
                value="name",
                options=sort_options)
controls = [text_input, filterby, sortby]
for control in controls:
     control.on_change('value', lambda attr, old, new: customer_directory_filter())


def customer_directory_filter():
    # sorting by date requires converting the character string in first_visit
    if (sortby.value == 'first_visit'):
        sql = "SELECT _id, name, email, phone_number, first_visit, TO_DATE(`first_visit`, 'MM/dd/yyyy') AS first_visit_date_type, churn_risk, sentiment FROM `dfs.default`.`./apps/crm` where " + str(filterby.value) +" like '%" + str(text_input.value.strip()) + "%' order by first_visit_date_type limit 10000"
    else:
        sql = "SELECT _id, name, email, phone_number, first_visit, churn_risk, sentiment FROM `dfs.default`.`./apps/crm` where " + str(filterby.value) +" like '%" + str(text_input.value.strip()) + "%' order by " + str(sortby.value) + " limit 10000"
    logger.debug("executing SQL: " + sql)
    global customer_directory_df, headshots, customer_directory_source
    customer_directory_df = pd.read_sql(sql, conn).dropna()
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
        'tenure': ((customer_directory_df.tenure / 365).astype(int)).astype(str) + 'yr',
        'email': customer_directory_df.email
    }

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
customer_directory_df.set_index('name')
customer_directory_df['headshot'] = headshots
customer_directory_df.loc[customer_directory_df['name'] == 'Eva Peterson', 'headshot'] = "84b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Richard Escalante', 'headshot'] = "1b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Joan Payne', 'headshot'] = "78b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Rachel Worrell', 'headshot'] = "79b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Nathan Porter', 'headshot'] = "54b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Anna Champlin', 'headshot'] = "28b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Erika Gallardo', 'headshot'] = "43b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Robert Macfarlane', 'headshot'] = "30b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Danny Rearick', 'headshot'] = "36b.jpg"
customer_directory_df.loc[customer_directory_df['name'] == 'Nicole Lewis', 'headshot'] = "26b.jpg"


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

    x = [purchases['Date'].iloc[0] + np.timedelta64(int(x.min() - 7), 'D'),
         purchases['Date'].iloc[0] + np.timedelta64(int(x.max() + 7), 'D')]
    y = [y_lr.min(), y_lr.max()]
    # lines_source = ColumnDataSource(data=dict(x=x, y=y))
    lines_source.data = dict(x=x, y=y)

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

columns = [
    TableColumn(field="name", title="Name", width=120),
    TableColumn(field="phone_number", title="Phone", width=100),
    TableColumn(field="tenure", title="Tenure", width=60, formatter=StringFormatter()),
    TableColumn(field="email", title="Email", width=150)
    # TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="0.000%")),
]

customer_directory_table = DataTable(source=customer_directory_source, columns=columns, row_headers=False,
                                     editable=True, width=280, height=300, fit_columns=False)

customer_directory_source.on_change('selected', lambda attr, old, new: selection_update(new))
customer_directory_source.data = {
    'name': customer_directory_df.name,
    'phone_number': customer_directory_df.phone_number,
    'tenure': ((customer_directory_df.tenure / 365).astype(int)).astype(str) + 'yr',
    'email': customer_directory_df.email
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
# Linear regression plot
################################

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

purchases['Day'] = purchases['Date'].apply(lambda x: (x - purchases['Date'].iloc[0]) / np.timedelta64(1, 'D')).astype(int)

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
plt.yaxis.formatter = NumeralTickFormatter(format="`$0,0`")
plt.circle('date', 'accumulated_purchases', source=purchase_history, size=2)
# plt.line([purchases['Date'].iloc[0] + np.timedelta64(x.min(), 'D'),
#           purchases['Date'].iloc[0] + np.timedelta64(x.max(), 'D')],
#          [y_lr.min(), y_lr.max()], color='red', line_width=2)
plt.xaxis.major_label_orientation = pi / 4

x = [purchases['Date'].iloc[0] + np.timedelta64(int(x.min() - 7), 'D'),
     purchases['Date'].iloc[0] + np.timedelta64(int(x.max() + 7), 'D')]
y = [y_lr.min(), y_lr.max()]
lines_source = ColumnDataSource(data=dict(x=x, y=y))
line = Line(x='x', y='y', line_color="red", line_width=2, line_alpha=.8)
plt.add_glyph(lines_source, line)

###############################################################################
# continuous animation graph for page views
#######################################
from numpy import asarray, cumprod, clip, ones, arange
from numpy.random import lognormal, rand, choice
from bokeh.layouts import column, row, gridplot, Spacer
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
spacer = Spacer(width=10)
row_control = row(filterby, spacer, sortby)
column1 = column(customer_directory_title, text_input, row_control, newline, customer_directory_table, query_performance, width=300)
column2 = column(widgetbox(ML_column_title, ML_table, width=300),
                 gridplot([[plt], [pageview_plt]], toolbar_location=None, plot_width=300))
column3 = widgetbox(Persona_column_title, headshot, selected_name, needs, has, width=300)
# drill_table_widget = widgetbox(drill_table)

curdoc().title = "Customer 360 Analytics"

headline_row = [headliner]
row1 = [title]
row2 = [column1, column2, column3]
row3 = [hm]
l1 = layout([row1, row2, row3], sizing_mode='fixed')
l2 = layout([headline_row], sizing_mode='fixed')
curdoc().add_root(l1)
curdoc().add_root(l2)
# curdoc().add_root(gridplot([[p2]], toolbar_location="left", plot_width=500))
curdoc().add_periodic_callback(cont_update, 100)
curdoc().title = "Customer 360 Demo"

make_default_selection()
