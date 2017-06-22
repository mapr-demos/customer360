slider = Slider(title="Annual Income", start=10000, end=100000, value=150000, step=1000)
slider.on_change('value', lambda attr, old, new: update())

autocomplete = AutocompleteInput(title="Name", completions=df['name'].tolist())
autocomplete.on_change('value', lambda attr, old, new: update())

button = Button(label="Download", button_type="success")
button.callback = CustomJS(args=dict(source=source),
                           code=open(join(dirname(__file__), "download.js")).read())

columns = [
    TableColumn(field="name", title="Employee Name"),
    TableColumn(field="salary", title="Income", formatter=NumberFormatter(format="$0,0.00")),
    TableColumn(field="years_experience", title="Experience (years)")
]

data_table = DataTable(source=source, columns=columns, row_headers=False)


pcolumns = [
    TableColumn(field="foo", title="Characteristic"),
    TableColumn(field="bar", title="Description")
]

psource = ColumnDataSource(data=dict(
    foo=['Telephone', 'Birthday'],
    bar=['327-623-5111', 'Nov 05 1978']))

pdata_table = DataTable(source=psource, columns=pcolumns, row_headers=False, width=200)

#
#
# from datetime import date
# from random import randint
#
# from bokeh.models import ColumnDataSource
# from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
#
#
# pdata = dict(
#         dates=[date(2014, 3, i+1) for i in range(10)],
#         downloads=[randint(0, 100) for i in range(10)],
#     )
# psource = ColumnDataSource(pdata)
#
# pcolumns = [
#         TableColumn(field="dates", title="Date", formatter=DateFormatter()),
#         TableColumn(field="downloads", title="Downloads")
#     ]
# pdata_table = DataTable(source=psource, columns=pcolumns, row_headers=False, width=200)
