import sys

sys.path.append('/usr/local/lib/python3.10/dist-packages/')
import mysql.connector
import openpyxl
import re
import pandas as pd





def write_alerts_data_into_db():
  conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="AlertBrain@1234"
  )
    # Create a cursor object to execute SQL queries
  cursor = conn.cursor()


  # Read the Excel file and specify the sheet name
  file_path = '/root/alertbrain/Alerts_Data1.xlsx'
  sheet_name = 'Sheet2'

  # Open the Excel file
  wb = openpyxl.load_workbook(file_path)

  # Get the second sheet
  sheet = wb[sheet_name]
  df = pd.DataFrame(columns=['category', 'message'], dtype= object)
  # Iterate over rows in the sheet
  for row in sheet.iter_rows(values_only=True):
      # Convert any None values to NULL
      row = [value if value is not None else 'NULL' for value in row]
      alerts = row[1].splitlines()
      #print(alerts)
      # Using a for loop
      for string in alerts:
          alert = re.sub(r"\b\d+\.", "", string, count=1)
          query = "INSERT INTO alert_brain.alert_mapping (category, message) VALUES (%s, %s)"
          values = (row[0], alert)
          df = df._append({'category': row[0], 'message': alert}, ignore_index=True)

          # Executing the query with the values
          cursor.execute(query, values)
          conn.commit()
  print(df)
  cursor.close()
  conn.close()




def read_all_alerts_from_db() :
    conn = mysql.connector.connect(
        host="localhost",
        user="root",
        password="AlertBrain@1234"
    )
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM alert_brain.alert_mapping")
    result = cursor.fetchall()
    cursor.close()
    conn.close()
    return result


