import mysql.connector as mysql
import bcrypt
import argparse
import sys

def encrypt_password(password):
    hash = bcrypt.hashpw(password,bcrypt.gensalt())
    return hash

def add_user(db, cursor):
    username = input("Enter user => ")
    password = input("Enter password => ").encode('utf-8')
    hashed_pw = encrypt_password(password)
    cursor.execute("insert into user_info (username, password) values (%s,%s)", (username, hashed_pw))
    db.commit()

def show_users(db, cursor):
    cursor.execute("select * from user_info")
    res = cursor.fetchall()
    if len(res) == 0:
        cursor.execute("alter table user_info auto_increment = 1")
        db.commit()
    for row in res:
        print(row)

def delete_user(db, cursor, user):
    confirmation = input("Are you sure? (Y/N) ")
    if confirmation == "Y":
        cursor.execute(f"delete from user_info where username = '{user}'")
        db.commit()
        print(cursor.rowcount, "record(s) deleted")
    else:
        return
    
parser = argparse.ArgumentParser(description="Sign up your user")
parser.add_argument('username',help="enter your database username",type=str)
parser.add_argument('password',help="enter your database password",type=str)

args = parser.parse_args()
username = args.username
password = args.password

db = mysql.connect(host="localhost", user=username, passwd=password, database="smart_system")
cursor = db.cursor()

while(True):
    try:
        print("Select option:\n1:Add users\n2:View user info\n3:Delete user\n")
        option = input("Option => ")
        if option == "1":
            add_user(db, cursor)
            print()
        if option == "2":
            show_users(db, cursor)
            print()
        if option == "3":
            user = input("Enter user => ")
            delete_user(db, cursor, user)
            print()

    except KeyboardInterrupt:
        sys.exit(0)
#print(username, password)

