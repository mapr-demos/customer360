"""
Notional Customer 360 Dashboard.
"""

import pandas as pd
import re
import logging
from subprocess import check_output
from os import listdir
from os.path import join, isfile
from bokeh.layouts import widgetbox, layout
from bokeh.models import ColumnDataSource, Div, Select, RadioGroup, Button
from bokeh.models.widgets import DataTable, TableColumn, StringFormatter, TextInput, DatePicker
from bokeh.models.widgets import Tabs, Panel
from bokeh.events import ButtonClick
from bokeh.themes import Theme
from fontawesome_icon import FontAwesomeIcon
import pyodbc
from bokeh.models import HoverTool
from datetime import datetime
import sys

# python3 code for adding to MapR-DB
ADD_CUST_SCRIPT = '/home/mapr/customer360/bokeh/add_cust.py'

# for a rotating selection of image files when adding a customer
hscnt = 0

# apply a theme
#theme = Theme(filename='./monokai_theme.yml')

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
intro = Div(text="""<h2><img src = "bokeh/static/images/topbanner.png" width="760" height="96"></h2>""")
customer_directory_title = Div(text="""<h3>Customer Directory:</h3>""")
ML_column_title = Div(text="""<h3>Customer Analysis:</h3>""")
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
newcust_intro = Div(text="""<h1>Add New Customer</h1><br>""")
campaigns_intro = Div(text="""<div class = "content"><h1>Marketing Campaigns</h1><br></div>""")
new_campaign_intro = Div(text="""<h3>New Customer Campaign:  Customers Eligible</h3>""")
newcust_complete = Div(text='')

sys.setrecursionlimit(10000)

logger = logging.getLogger('bokeh')
logger.setLevel(logging.DEBUG)
customer_directory_df = pd.DataFrame()
campaign_directory_df = pd.DataFrame()
conn = pyodbc.connect("DSN=drill64", autocommit=True)

def crm_table_update():
    global customer_directory_df
    global conn
    # Specify unicode options only for MapR 5.2:
    # conn.setdecoding(pyodbc.SQL_CHAR, encoding='utf-32le', to=str)
    # conn.setdecoding(pyodbc.SQL_WMETADATA, encoding='utf-32le', to=str)
    
    cursor = conn.cursor()
    
    sql = "SELECT _id, name, address, email, phone_number, \
          latitude, longitude, first_visit, churn_risk, sentiment \
          FROM `dfs.default`.`./tmp/crm_data` limit 10000"
    logger.debug("executing SQL: " + sql)
    customer_directory_df = pd.read_sql(sql, conn)
    logger.debug("records returned: " + str(len(customer_directory_df.index)))
    query_performance.text = "<div class=\"small\">" + \
        str(len(customer_directory_df.index)) + " rows selected</div>"
    
def newcust_add_callback(event):
    gendstr = gendopts[newcust_gender.active]
    bdatestr = newcust_birthdate.value.strftime('%m/%d/%Y')
    fvdatestr = newcust_firstvisit.value.strftime('%m/%d/%Y')
    print("New customer added")
    print("name: " + newcust_name.value)
    print("gender: " + gendstr)
    print("address: " + newcust_address.value)
    print("zip: " + newcust_zip.value)
    print("phone: " + newcust_phone.value)
    print("ssn: " + newcust_ssn.value)
    print("email: " + newcust_email.value)
    print("firstvisit: " + fvdatestr)
    print("birthdate: " + bdatestr)
    print("notes: " + newcust_notes.value)
    out = check_output([ADD_CUST_SCRIPT, newcust_name.value, gendstr,
        newcust_address.value, newcust_state.value, newcust_ssn.value, newcust_zip.value,
        newcust_email.value, newcust_phone.value, fvdatestr,
        bdatestr, "POSITIVE", "1", "1"])
    print("tool output is: %s" % out)
    if (re.match('[a-z0-9]*-[a-z0-9]*-[a-z0-9]*', out)):
        print("ID assigned:  %s" % out)
        newcust_complete.text = \
            '<img src="bokeh/static/images/opsuccess.png" \
                 width="40" align="middle" hspace="8">' + \
                '<b>Successfully added.  ID assigned:  %s</b>' % out

def newcust_attachphoto_callback(event):
    global hscnt
    hscnt += 1
    newcust_attachshot.text = str('<img src=') + newcust_hs_path + \
        newcust_avail_shots[hscnt % len(newcust_avail_shots)] + str(' alt="face" width="70">')
    print("attach_photo called")
    print("new text is: " + newcust_attachshot.text)

def testcb(attr, old, new):
    print("completed attr: " + attr)
    print("prev label " + old)
    print("new label " + new)

# grab the first set of customer data
crm_table_update()

newcust_name = TextInput(title='Customer Name', value='', width=150)
newcust_name.on_change("value", testcb)
gendopts = [ 'MALE', 'FEMALE', 'OTHER' ]
newcust_gender_label = Div(text='Gender')
newcust_gender = RadioGroup(name='Gender', labels=gendopts, active=0)
newcust_address = TextInput(title='Address', value='', width=150)
stateopts = [ "AK", "AL", "AR", "AZ", "CA", "CO", "CT", "DC",
    "DE", "FL", "GA", "HI", "IA", "ID", "IL", "IN", "KS", "KY", "LA",
    "MA", "MD", "ME", "MI", "MN", "MO", "MS", "MT", "NC", "ND", "NE",
    "NH", "NJ", "NM", "NV", "NY", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VA", "VT", "WA", "WI", "WV", "WY" ]
newcust_state = Select(title='State', width=200, value='AK', options=stateopts)
newcust_ssn = TextInput(title='SSN', value='', width=150)
newcust_zip = TextInput(title='ZIP', value='', width=150)
newcust_email = TextInput(title='Email', value='', width=150)
newcust_dob = TextInput(title='DOB', value='', width=150)
newcust_phone = TextInput(title='Phone', value='', width=150)
newcust_firstvisit = DatePicker(title='First contact', value='09/18/2017', width=150)
leadsources = [ 'Spring17Campaign', 'Web', 'MailCampaign#1', 'Walk-in', 'Referral', 'Other' ]
newcust_leadsource = Select(title='Lead source', width=182, value='Web', options=leadsources)
newcust_birthdate = DatePicker(title='Birthdate', value='01/01/1980', width=150)
newcust_notes = TextInput(title='Notes', value='', width=150, height=50)
newcust_notes.on_change("value", testcb)
newcust_attach_button = Button(label='Attach Photo', button_type="success")
newcust_attach_button.on_event(ButtonClick, newcust_attachphoto_callback)
newcust_add_button = Button(label='Add Customer', button_type="success")
newcust_add_button.on_event(ButtonClick, newcust_add_callback)
newcust_attachshot = Div(text='<img src="bokeh/static/face_images/generic_hs.png" \
    alt="face" width="110">')

newcust_hs_path = "bokeh/static/face_images/"

newcust_avail_shots = [ "166b.jpg", "158b.jpg", "140b.jpg", "97b.jpg" ]

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
        sql = "SELECT _id, name, email, phone_number, first_visit, TO_DATE(`first_visit`, 'MM/dd/yyyy') AS first_visit_date_type, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` where " + filterby.value +" like '%" + text_input.value.strip() + "%' order by first_visit_date_type limit 10000"
    else:
        sql = "SELECT _id, name, email, phone_number, first_visit, churn_risk, sentiment FROM `dfs.default`.`./tmp/crm_data` where " + filterby.value +" like '%" + text_input.value.strip() + "%' order by " + sortby.value + " limit 10000"
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
campaign_directory_source = ColumnDataSource(data=dict())

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

cmp_columns = [
    TableColumn(field="name", title="Name", width=150),
    TableColumn(field="address", title="Address", width=200),
    TableColumn(field="state", title="State", width=75, formatter=StringFormatter()),
    TableColumn(field="email", title="Email", width=200),
    TableColumn(field="phone_number", title="Phone", width=125),
    TableColumn(field="persona", title="Persona", width=30),
    TableColumn(field="sentiment", title="Sentiment", width=120),
]

customer_directory_table = DataTable(source=customer_directory_source, columns=columns, row_headers=False,
                                     editable=True, width=280, height=300, fit_columns=False)
campaign_directory_table = DataTable(source=campaign_directory_source, columns=cmp_columns, row_headers=False,
                                     editable=True, width=900, height=300, fit_columns=False)

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

def campaign_table_update():
    global campaign_directory_df
    global campaign_directory_table
    global campaign_directory_source
    global conn
    
    cursor = conn.cursor()
    
    sql = "SELECT _id, name, address, email, phone_number, state, persona, zip, \
          latitude, longitude, first_visit, churn_risk, sentiment \
          FROM `dfs.default`.`./tmp/crm_data` order by phone_number limit 100"
    logger.debug("executing SQL: " + sql)
    campaign_directory_df = pd.read_sql(sql, conn)
    campaign_directory_source.data = {
        'name': campaign_directory_df.name,
        'address': campaign_directory_df.address,
        'zip': campaign_directory_df.zip,
        'state': campaign_directory_df.state,
        'email': campaign_directory_df.email,
        'phone_number': campaign_directory_df.phone_number,
        'persona': campaign_directory_df.persona,
        'sentiment': campaign_directory_df.sentiment,
    }
    campaign_directory_table.source = campaign_directory_source
    campaign_directory_table.update()
    logger.debug("(campaign table) records returned: " + str(len(campaign_directory_df.index)))

# update the table for the first time
campaign_table_update()

##############################################################################
# Create heatmap
##########################

from heatmap import create_heatmap

hm = create_heatmap(customer_directory_df)


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
plt = figure(width=280, height=200, x_axis_type="datetime", title='Spend Rate Analysis',
    toolbar_location=None)
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

pageview_plt = figure(plot_height=150, x_axis_location=None, y_axis_location="left", title='Clickstream')
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
column3 = widgetbox(Persona_column_title, headshot, selected_name, needs, has, width=200)
# drill_table_widget = widgetbox(drill_table)

newcust_column1 = column(newcust_name, newcust_gender_label, newcust_gender,
    newcust_address, width=200)
newcust_column2 = column(newcust_zip, newcust_phone, newcust_ssn, newcust_email,
    width=200)
newcust_column3 = column(newcust_firstvisit, newcust_leadsource, newcust_birthdate, width=200)
newcust_column4 = column(newcust_attachshot,
    newcust_attach_button, newcust_add_button, width=200)
newcust_title = widgetbox(newcust_intro, width=800)
newcust_completetitle = widgetbox(newcust_complete, width=800)

campaigns_title = widgetbox(campaigns_intro, width=800)
new_campaigns_title = widgetbox(new_campaign_intro, width=800)
campaigns_column = column(campaign_directory_table, width=800)
campaigns_fb_button = \
    Button(icon=FontAwesomeIcon(icon_name='facebook-square', size=2), \
    label=' Run Facebook Campaign', button_type="success")
campaigns_twitter_button = \
    Button(icon=FontAwesomeIcon(icon_name='twitter-square', size=2), \
    label=' Run Twitter Campaign', button_type="success")
campaigns_linkedin_button = \
    Button(icon=FontAwesomeIcon(icon_name='linkedin-square', size=2), \
    label=' Run LinkedIn Campaign', button_type="success")
campaigns_email_button = \
    Button(icon=FontAwesomeIcon(icon_name='pencil-square', size=2), \
    label=' Run Email Campaign', button_type="success")
newcust_attach_button = Button(label='Attach Photo', button_type="success")

curdoc().title = "Customer 360 Analytics"
#curdoc().theme = theme

row1 = [title]
row2 = [column1, column2, column3 ]
row3 = [hm]
row4 = [newcust_title]
row5 = [ newcust_column1, newcust_column2, newcust_column3, newcust_column4 ]
row6 = [ newcust_completetitle ]
row7 = [ campaigns_title ]
row8 = [ new_campaigns_title ]
row9 = [ campaigns_column ]
row10 = [ campaigns_fb_button, campaigns_twitter_button ]
row11 = [ campaigns_linkedin_button, campaigns_email_button ]
l1 = layout([ row1, row2, row3 ], sizing_mode='fixed')
l2 = layout([ row4, row5, row6 ], sizing_mode='fixed')
l3 = layout([ row7, row8, row9, row10, row11 ], sizing_mode='fixed')
tab1 = Panel(child=l1,title="Retail Analytics Dashboard")
tab2 = Panel(child=l2,title="Add New Customer")
tab3 = Panel(child=l3,title="Marketing Campaigns Dashboard")
tabs = Tabs(tabs=[ tab1, tab2, tab3 ])

curdoc().add_root(tabs)
#curdoc().add_root(l1)
#curdoc().add_root(l2)
# curdoc().add_root(gridplot([[p2]], toolbar_location="left", plot_width=500))
curdoc().add_periodic_callback(cont_update, 100)
#curdoc().add_periodic_callback(crm_table_update, 10000)
curdoc().add_periodic_callback(campaign_table_update, 5000)
curdoc().title = "Customer 360 Demo"

make_default_selection()
