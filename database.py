import sqlite3 
  
# filename to form database 
file = "user_info.db"
  
try: 
  conn = sqlite3.connect(file) 
  print("Database user_info.db formed.") 
except: 
  print("Database user_info.db not formed.")

cursor = conn.cursor()

sql ='''CREATE TABLE USERS(
   USER_NAME VARCHAR(255) PRIMARY KEY,
   PASSWORD VARCHAR(255) NOT NULL,
   CITY VARCHAR(255) NOT NULL
)'''

cursor.execute(sql)

# Commit your changes in the database
conn.commit()

#Closing the connection
conn.close()