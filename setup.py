from sqlalchemy import create_engine

# Define the database URI (can be dynamically set by the user)
db_uri = 'mssql+pyodbc://@NAMAN\SQLEXPRESS01/SQL(basic to advance)?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes'  # Example: 'postgresql://user:password@localhost/mydatabase'

# Create an engine to connect to the database
engine = create_engine(db_uri)

# Test the connection by running a simple query (optional)
with engine.connect() as connection:
    result = connection.execute("SELECT 1")
    for row in result:
        print(row)

# Now you can use this engine to run queries on the database the user has connected to