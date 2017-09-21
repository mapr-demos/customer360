import pyodbc
from pandas import *

# Initialize the connection
# The DSN was defined with the iODBC Administrator app for Mac.
conn = pyodbc.connect("DSN=drill64", autocommit=True)
cursor = conn.cursor()

# Setup a SQL query to select data from a csv file.
# The csv2 filename extension tells Drill to extract
# column names from the first row.
s = "SELECT * FROM `dfs.tmp`.`./companylist.csv2` limit 3"

# Execute the SQL query
print(pandas.read_sql(s, conn))

# Here's how to select data from MySQL
s = "select * from ianmysql.mysql.`user`"
pandas.read_sql(s, conn)

# Here's an example of a SQL JOIN the combines a JSON file with a MySQL table.
s = "SELECT tbl1.name, tbl2.address FROM `dfs.tmp`.`./names.json` as tbl1 \
    JOIN `dfs.tmp`.`./addressunitedstates.json` as tbl2 ON tbl1.id=tbl2.id"
pandas.read_sql(s, conn)

