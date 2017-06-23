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
from bokeh.models import ColumnDataSource, CustomJS, FixedTicker, Div, PreText, Range1d, Select
from bokeh.io import curdoc
from bokeh.models.widgets import Slider, Button, DataTable, TableColumn, NumberFormatter, TextInput, AutocompleteInput
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
        'Prediction': [str(int(current['churn_risk'].iloc[0]))+'%',
                       current['sentiment'].iloc[0],
                       'Group ' + str(np.random.randint(low=1,high=4)),
                       'Auto Loan',
                       '$' + str(500 + np.random.randint(low=0, high=499)) + ',' + str(100 + np.random.randint(low=0, high=899))]
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
        'Prediction': [str(int(current['churn_risk'].iloc[0]/100)+1)+'%',
                       current['sentiment'].iloc[0],
                       'Group ' + str(np.random.randint(low=1,high=4)),
                       'Auto Loan',
                       '$' + str(500 + np.random.randint(low=0, high=499)) + ',' + str(100 + np.random.randint(low=0, high=899))]
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
                       str(int(current['churn_risk'].iloc[0]/100))+'%',
                       current['sentiment'].iloc[0],
                       'Group ' + str(np.random.randint(low=1,high=4)),
                       '$' + str(500 + np.random.randint(low=0, high=499)) + ',' + str(100 + np.random.randint(low=0, high=899))]
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

customer_directory_table = DataTable(source=source, columns=columns, row_headers=False, editable=True, width=280, height=480)
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

ML_table = DataTable(source=machine_learning_table, columns=churn_table_columns, row_headers=False, editable=True, width=280, height=160)

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



###############################################################################
# continuous animation graph
#######################################
from numpy import asarray, cumprod, convolve, exp, ones
from numpy.random import lognormal, gamma, uniform
from bokeh.layouts import row, column, gridplot
from bokeh.plotting import curdoc, figure
from bokeh.driving import count

BUFSIZE = 200
MA12, MA26, EMA12, EMA26 = '12-tick Moving Avg', '26-tick Moving Avg', '12-tick EMA', '26-tick EMA'

cont_source = ColumnDataSource(dict(
    time=[], average=[], low=[], high=[], open=[], close=[],
    ma=[], macd=[], macd9=[], macdh=[], color=[]
))

p = figure(plot_height=250, x_axis_type=None, y_axis_location="right")

p.line(x='time', y='average', alpha=0.2, line_width=3, color='navy', source=cont_source)
p.line(x='time', y='ma', alpha=0.8, line_width=2, color='orange', source=cont_source)
p.segment(x0='time', y0='low', x1='time', y1='high', line_width=2, color='black', source=cont_source)
p.segment(x0='time', y0='open', x1='time', y1='close', line_width=8, color='color', source=cont_source)

p2 = figure(plot_height=150, x_range=p.x_range, y_axis_location="left", title='Support Ticket Volume')
p2.line(x='time', y='macd', line_width=2, source=cont_source)
p2.x_range.follow = "end"
p2.x_range.follow_interval = 100
p2.x_range.range_padding = 0


mean = Slider(title="mean", value=0, start=-0.01, end=0.01, step=0.001)
stddev = Slider(title="stddev", value=0.04, start=0.01, end=0.1, step=0.01)
mavg = Select(value=MA12, options=[MA12, MA26, EMA12, EMA26])

def _create_prices(t):
    last_average = 100 if t==0 else cont_source.data['average'][-1]
    returns = asarray(lognormal(mean.value, stddev.value, 10))
    average = last_average * cumprod(returns)
    high = average * exp(abs(gamma(1, 0.03, size=1)))
    low = average / exp(abs(gamma(1, 0.03, size=1)))
    delta = high - low
    open = low + delta * uniform(0.05, 0.95, size=1)
    close = low + delta * uniform(0.05, 0.95, size=1)
    return open[0], high[0], low[0], close[0], average[0]

def _moving_avg(prices, days=10):
    if len(prices) < days: return [100]
    return convolve(prices[-days:], ones(days, dtype=float), mode="valid") / days

def _ema(prices, days=10):
    if len(prices) < days or days < 2: return [prices[-1]]
    a = 2.0 / (days+1)
    kernel = ones(days, dtype=float)
    kernel[1:] = 1 - a
    kernel = a * cumprod(kernel)
    # The 0.8647 normalizes out that we stop the EMA after a finite number of terms
    return convolve(prices[-days:], kernel, mode="valid") / (0.8647)

@count()
def cont_update(t):
    open, high, low, close, average = _create_prices(t)
    color = "green" if open < close else "red"

    new_data = dict(
        time=[t],
        open=[open],
        high=[high],
        low=[low],
        close=[close],
        average=[average],
        color=[color],
    )

    close = cont_source.data['close'] + [close]
    ma12 = _moving_avg(close[-12:], 12)[0]
    ma26 = _moving_avg(close[-26:], 26)[0]
    ema12 = _ema(close[-12:], 12)[0]
    ema26 = _ema(close[-26:], 26)[0]

    if   mavg.value == MA12:  new_data['ma'] = [ma12]
    elif mavg.value == MA26:  new_data['ma'] = [ma26]
    elif mavg.value == EMA12: new_data['ma'] = [ema12]
    elif mavg.value == EMA26: new_data['ma'] = [ema26]

    macd = ema12 - ema26
    new_data['macd'] = [macd]

    macd_series = cont_source.data['macd'] + [macd]
    macd9 = _ema(macd_series[-26:], 26)[0]
    new_data['macd9'] = [macd9]
    new_data['macdh'] = [macd - macd9]

    cont_source.stream(new_data, 300)

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
column2 = column(widgetbox(ML_column_title, ML_table, width=300), gridplot([[plt], [p2]], toolbar_location=None, plot_width=300))
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
# curdoc().add_root(gridplot([[p2]], toolbar_location="left", plot_width=500))
curdoc().add_periodic_callback(cont_update, 100)
curdoc().title = "MapR Customer 360 Demo"

make_default_selection()


