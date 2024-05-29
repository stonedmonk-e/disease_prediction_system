import sqlite3



def correct_login(username, password):
    return check_user(username, password)

def correct_signup(username, password, confirm_password, city):
    if(password==confirm_password): 
        if(add_user(username, password, city)):
            return True
        else:
            return False
    else:
        return False
    
    
def add_user(username, password, city):
    conn = sqlite3.connect("user_info.db")
    cursor = conn.cursor()
    #print(f"SELECT * FROM users WHERE user_name ='{username}'")
    cursor.execute(f"SELECT 1 FROM users WHERE user_name = '{username}'")
    row = cursor.fetchone()

    # If a row is found, the username exists
    if row is not None:
        conn.commit()
        conn.close()
        return False
    
    else:
        sql = f""" INSERT INTO USERS 
        (USER_NAME, PASSWORD,CITY) 
        VALUES ('{username}', '{password}','{city}')"""
        cursor.execute(sql)
        conn.commit()
        conn.close()
        return True

def check_user(username, password):
    conn = sqlite3.connect("user_info.db")
    cursor = conn.cursor()
    cursor.execute(f"SELECT password FROM users WHERE user_name = '{username}'")
    row = cursor.fetchone()
    
    if row == None:
        return False
    conn.commit()
    conn.close()
    stored_password = row[0]
    if password == stored_password:
        return True
    else:
        return False