from fastapi import FastAPI
import pymysql


def get_db_connection():
    connection = pymysql.connect(
        host="ithost.pongsawadi.ac.th",
        user="ithost68ry1cibzpuo3p",
        password="2r5KgpUfG!5hWkJjT!!3CSFg",
        database="ithost68ry1cibzpuo3p"
    )
    return connection

app = FastAPI()

@app.get("/test/attractions")
def read_attractions():
    connection = get_db_connection()
    cursor = connection.cursor()
    query = "SELECT * FROM attractions"
    cursor.execute(query)
    row = cursor.fetchall()
    cursor.close()
    connection.close()

    attractions = []
    for row in row:
        attractions.append({
            "id": row[0],
            "name": row[1],
            "detail": row[2],
            "coverimage": row[3],
        })
    return attractions

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/test")
def read_test():
    return {"message": "Hello from /test"}

@app.get("/test/{id}")
def read_testId(id: int):
    return {"message": f"Hello from /test/{id}"}