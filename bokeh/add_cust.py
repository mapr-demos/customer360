#!/usr/bin/python3

import maprdb
import sys
import random
from numpy import int32
import os

TABLEPATH = '/tmp/crm_data'

if (len(sys.argv) != 14):
    print("incorrect number of args:  arglen is %d" % len(sys.argv))
    print("usage:  %s <name> <gender> <address> <state> <ssn> <zip> <email> <phone> <first_visit>"
            " <birth_date> <sentiment> <churn_risk> <persona>")
    sys.exit(0)

name = sys.argv[1]
gender = sys.argv[2]
address = sys.argv[3]
state = sys.argv[4]
ssn = sys.argv[5]
zzip = sys.argv[6]
latitude = "0.00"
longitude = "0.00"
email = sys.argv[7]
phone_number = sys.argv[8]
first_visit = sys.argv[9]
birth_date = sys.argv[10]
sentiment = sys.argv[11]
churn_risk = sys.argv[12]
persona = sys.argv[13]


# XXX is there a way to make the API pick this for us
# XXX for a new add
idkey = '%08x-%04x-%04x-%04x-%012x' %  \
    (random.randrange(16**8), random.randrange(16**4),
     random.randrange(16**4), random.randrange(16**4),
     random.randrange(16**12))

# for now just call the shell, this does it correctly
cmd = 'mapr dbshell "insert /tmp/crm_data --id %s --value \
    \'{\\"address\\":\\"%s\\", \
    \\"first_visit\\":\\"%s\\",\\"persona\\":%s,\\"ssn\\":\\"%s\\", \
    \\"churn_risk\\":%s,\\"email\\":\\"%s\\", \
    \\"sentiment\\":\\"%s\\",\\"gender\\":\\"%s\\",\\"phone_number\\":\\"%s\\", \
    \\"zip\\":\\"%s\\",\\"name\\":\\"%s\\", \
    \\"latitude\\":\\"%s\\",\\"birth_date\\":\\"%s\\",\\"state\\":\\"%s\\",\\"longitude\\":\\"%s\\" }\'"' % \
    (idkey, address, first_visit, persona, ssn, churn_risk, email, sentiment,
        gender, phone_number, zzip, name, "0.00", birth_date, state, "0.00")
finalcmd = cmd + " > /dev/null 2>&1"
#print(finalcmd)
os.system(finalcmd)

# XXX below is the OJAI version of this script, cannot be used
# XXX at the moment because 'churn' and 'persona' cause the python-bindings
# XXX to add it as a weird '{"$numberLong":1}' type, and when I use numpy's int32()
# XXX it causes an unknown binding error.  need to look into this more.

#
# c = maprdb.connect()
# t = c.get(TABLEPATH)
# 
# newdoc = maprdb.Document(
#     {
#     '_id' : idkey,
#     'name': name,
#     'gender' : gender,
#     'address' : address,
#     'state' : state,
#     'ssn' : ssn,
#     'zip' : zzip,
#     'latitude' : latitude,
#     'longitude' : longitude,
#     'email' : email,
#     'phone_number' : phone_number,
#     'first_visit' : first_visit,
#     'birth_date' : birth_date,
#     'sentiment' : sentiment,
#     'churn_risk' : int32(1),
#     'persona' : int32(1),
#     })
# 
# t.insert_or_replace(newdoc)
# t.close()

print("%s" % idkey)
