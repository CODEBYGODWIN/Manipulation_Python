import sqlite3
import pandas as pd

conn = sqlite3.connect('Chinook_SQLite.sqlite')
cursor = conn.cursor()

cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()

query = '''
SELECT BillingCountry, SUM(Total) as Revenue
FROM invoice
GROUP BY BillingCountry
ORDER BY Revenue DESC
LIMIT 10;
'''
df = pd.read_sql_query(query, conn)
print(df)

# Quels sont les 5 clients ayant dépensé le plus ?
query = '''
SELECT Customer.FirstName, Customer.LastName, SUM(Invoice.Total) as TotalSpent
FROM Customer
JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
GROUP BY Customer.CustomerId
ORDER BY TotalSpent DESC
LIMIT 5;
'''
df2 = pd.read_sql_query(query, conn)
print(df2)

#Quels genres musicaux rapportent le plus ?
query = '''
SELECT Genre.Name, SUM(InvoiceLine.UnitPrice * InvoiceLine.Quantity) as Revenue
FROM Genre
JOIN Track ON Genre.GenreId = Track.GenreId
JOIN InvoiceLine ON Track.TrackId = InvoiceLine.TrackId
JOIN Invoice ON InvoiceLine.InvoiceId = Invoice.InvoiceId
GROUP BY Genre.GenreId
ORDER BY Revenue DESC
LIMIT 5;
'''
df3 = pd.read_sql_query(query, conn)
print(df3)
#Quelle est la durée moyenne d’un morceau de Rock ?
query = '''
SELECT AVG(Track.Milliseconds) as AvgDuration
FROM Genre
JOIN Track ON Genre.GenreId = Track.GenreId
WHERE Genre.Name = 'Rock';
'''
df4 = pd.read_sql_query(query, conn)
print(df4)

#Quel employé (Sales Support Agent) a généré le plus de revenus ?
query = '''
SELECT Employee.FirstName, Employee.LastName, SUM(Invoice.Total) as TotalRevenue
FROM Employee
JOIN Customer ON Employee.EmployeeId = Customer.SupportRepId
JOIN Invoice ON Customer.CustomerId = Invoice.CustomerId
WHERE Employee.Title = 'Sales Support Agent'
GROUP BY Employee.EmployeeId
ORDER BY TotalRevenue DESC
LIMIT 1;
'''
df5 = pd.read_sql_query(query, conn)
print(df5)
conn.close()