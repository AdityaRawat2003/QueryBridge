from dotenv import load_dotenv
from transformers import T5Tokenizer, T5ForConditionalGeneration
from sqlalchemy import create_engine, text
import streamlit as st
import sentencepiece
import torch
import os

# Ensure the required packages path is in Python's sys.path
import sys
sys.path.append("D:/Major Project clone/venv/Lib/site-packages")

# Load environment variables
load_dotenv()

# Set up the model and tokenizer paths and load them
tokenizer = T5Tokenizer.from_pretrained("t5-small", cache_dir="./model", force_download=True)
model = T5ForConditionalGeneration.from_pretrained("t5-small", cache_dir="./model", force_download=True)

# Function to generate SQL queries from natural language questions
def generate_sql_query(question):
    input_text = prompt[0] + "\n" + question  # Concatenate prompt with question
    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**inputs)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return sql_query

# Function to execute SQL queries and retrieve results
def read_sql_query(sql, db_uri):
    engine = create_engine(db_uri)
    with engine.connect() as connection:
        result = connection.execute(text(sql))
        rows = result.fetchall()
    return rows

# Prompt for model
prompt = [
    '''
    You are an expert in converting English questions to SQL query!
    Refer all the tables from a database (wherever there are more than 1 table in a database).
    A database can have many tables. You need to see from which table the query can be best executed.
    In some cases, you also need to join the tables. 
    For example,
    Example 1 - How many employees have a salary more than 45000?
    SQL command: SELECT * FROM ED JOIN ES ON ED.EmployeeID = ES.EmployeeID WHERE Salary > 45000;
    
    Example 2 - Tell me all the students studying in Data Science class?
    SQL command: SELECT * FROM STUDENT WHERE CLASS LIKE 'Data Science';
    
    Match the SQL database name. If not, then show an error.
    Avoid using ``` and the word "sql" in the output.
    Do not match the case sensitivity.
    '''
]

# Streamlit configuration
st.set_page_config(page_title="Text-to-SQL Converter", layout="wide")

# Sidebar for user inputs
st.sidebar.title("Query Builder")
st.sidebar.write("Convert natural language to SQL and fetch data from the database.")
question = st.sidebar.text_input("Enter your query:", key="input")

# Dropdown for selecting the database type
db_type = st.sidebar.selectbox("Select the database type:", ["SQLite", "MySQL", "PostgreSQL", "Microsoft SQL Server"])

# Display URI format for the selected database type
if db_type == "SQLite":
    st.sidebar.write("URI format: sqlite:///path/to/database.db")
    db_uri = st.sidebar.text_input("Enter the database URI:", key="db_input", value="sqlite:///student.db")
elif db_type == "MySQL":
    st.sidebar.write("URI format: mysql+pymysql://username:password@host:port/database")
    db_uri = st.sidebar.text_input("Enter the database URI:", key="db_input", value="mysql+pymysql://username:password@localhost:3306/database_name")
elif db_type == "PostgreSQL":
    st.sidebar.write("URI format: postgresql+psycopg2://username:password@host:port/database")
    db_uri = st.sidebar.text_input("Enter the database URI:", key="db_input", value="postgresql+psycopg2://username:password@localhost:5432/database_name")
elif db_type == "Microsoft SQL Server":
    st.sidebar.write("URI format: mssql+pyodbc://@server_name/database_name?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes")
    db_uri = st.sidebar.text_input("Enter the database URI:", key="db_input", value="mssql+pyodbc://@server_name/database_name?driver=ODBC+Driver+18+for+SQL+Server&trusted_connection=yes")

submit = st.sidebar.button("Generate Results")

# Initialize session state
if 'generated_sql' not in st.session_state:
    st.session_state.generated_sql = None
if 'show_sql' not in st.session_state:
    st.session_state.show_sql = False

# Main page layout
st.title("🧠 QueryBridge")
st.subheader("Effortlessly convert your natural language questions into SQL queries and get data.")

if submit and question:
    # Generate SQL query using the model
    generated_sql = generate_sql_query(question)
    st.session_state.generated_sql = generated_sql

    # Display results based on the generated SQL
    st.subheader("Query Results")
    try:
        results = read_sql_query(st.session_state.generated_sql, db_uri)
        if results:
            # Display results in a structured format
            headers = results[0].keys()  # Assuming results is a list of dictionaries
            st.write(headers)  # Display headers
            for row in results:
                st.write(row)  # Display each row as a dictionary
        else:
            st.write("No data found for the query.")
    except Exception as e:
        st.error(f"An error occurred while executing the query: {e}")

# Sidebar button to show or hide the SQL query
if st.session_state.generated_sql:
    if st.sidebar.button('Show SQL'):
        st.session_state.show_sql = not st.session_state.show_sql

# Conditionally show the generated SQL query
if st.session_state.show_sql and st.session_state.generated_sql:
    st.subheader("Generated SQL Query")
    st.code(st.session_state.generated_sql, language='sql')
